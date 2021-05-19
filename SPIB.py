"""
SPIB: A deep learning-based framework to learn RCs 
from MD trajectories. Code maintained by Dedi.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/5.0038198
"""
import torch
from torch import nn
import numpy as np
import os
import torch.nn.functional as F
        
# --------------------
# Model
# --------------------   

class SPIB(nn.Module):

    def __init__(self, z_dim, pseudo_dim, output_dim, data_shape, device, UpdateLabel= False, neuron_num1=128, 
                 neuron_num2=128):
        
        super(SPIB, self).__init__()
        
        self.z_dim = z_dim
        self.pseudo_dim = pseudo_dim
        self.output_dim = output_dim
        
        self.neuron_num1 = neuron_num1
        self.neuron_num2 = neuron_num2
        
        self.data_shape = data_shape
        
        self.UpdateLabel = UpdateLabel
        
        self.eps = 1e-10
        self.device = device
        
        # create an idle input for calling pseudo-inputs
        # torch buffer, these variables will not be trained
        self.idle_input = torch.eye(self.pseudo_dim, self.pseudo_dim, device=device, requires_grad=False)
        
        # pseudo weights
        self.pseudo_weights = nn.Sequential(
            nn.Linear(self.pseudo_dim, 1, bias=False),
            nn.Softmax(dim=0))
        
        self.pseudo_means = nn.Linear(self.pseudo_dim, np.prod(self.data_shape), bias=False)
        
        self.encoder = self._encoder_init()
        
        self.encoder_mean = nn.Linear(self.neuron_num1, self.z_dim)
        
        # enforce log_var in the range of [-10, 0]
        self.encoder_logvar = nn.Sequential(
            nn.Linear(self.neuron_num1, self.z_dim),
            nn.Sigmoid())
        
        self.decoder = self._decoder_init()
        
    def _encoder_init(self):
        
        modules = [nn.Linear(np.prod(self.data_shape), self.neuron_num1)]
        modules += [nn.ReLU()]
        for _ in range(1):
            modules += [nn.Linear(self.neuron_num1, self.neuron_num1)]
            modules += [nn.ReLU()]
        
        return nn.Sequential(*modules)
    
    def _decoder_init(self):
        # cross-entropy MLP decoder
        # output the probability of future state
        modules = [nn.Linear(self.z_dim, self.neuron_num2)]
        modules += [nn.ReLU()]
        for _ in range(1):
            modules += [nn.Linear(self.neuron_num2, self.neuron_num2)]
            modules += [nn.ReLU()]
        
        modules += [nn.Linear(self.neuron_num2, self.output_dim)]
        modules += [nn.LogSoftmax(dim=1)]
        
        return nn.Sequential(*modules)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def encode(self, inputs):
        enc = self.encoder(inputs)
        
        z_mean = self.encoder_mean(enc)
        # enforce log_var in the range of [-10, 0]
        z_logvar = -10*self.encoder_logvar(enc)
        
        return z_mean, z_logvar
    
    def forward(self, data):
        inputs = torch.flatten(data, start_dim=1)
        
        z_mean, z_logvar = self.encode(inputs)
        
        z_sample = self.reparameterize(z_mean, z_logvar)
        
        outputs = self.decoder(z_sample)
        
        return outputs, z_sample, z_mean, z_logvar
    
    def log_p (self, z, sum_up=True):
        # get pseudo_z - pseudo_dim * z_dim
        pseudo_z_mean, pseudo_z_logvar = self.get_pseudo_z()
        # get pseudo weights - pseudo_dim * 1
        w = self.pseudo_weights(self.idle_input)
        # w = 0.5*torch.ones((2,1)).to(self.device)
        
        # expand z - batch_size * z_dim
        z_expand = z.unsqueeze(1)
        
        pseudo_mean = pseudo_z_mean.unsqueeze(0)
        pseudo_logvar = pseudo_z_logvar.unsqueeze(0)
        
        # pseudo log_q
        pseudo_log_q = -0.5 * torch.sum(pseudo_logvar + torch.pow(z_expand-pseudo_mean, 2)
                                        / torch.exp(pseudo_logvar), dim=2 )
        
        if sum_up:
            log_p = torch.sum(torch.log(torch.exp(pseudo_log_q)@w + self.eps), dim=1)
        else:
            log_p = torch.log(torch.exp(pseudo_log_q)*w.T + self.eps)  
            
        return log_p
        
    # the prior
    def get_pseudo_z(self):
        # calculate pseudo_means
        # with torch.no_grad():
        X = self.pseudo_means(self.idle_input)

        # calculate pseudo_z
        pseudo_z_mean, pseudo_z_logvar = self.encode(X)  # C x M

        return pseudo_z_mean, pseudo_z_logvar
            
    @torch.no_grad()
    def update_labels(self, inputs, batch_size):
        if self.UpdateLabel:
            labels = []
            
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i+batch_size].to(self.device)
            
                # pass through VAE
                z_mean, z_logvar = self.encode(batch_inputs)        
                log_prediction = self.decoder(z_mean)
                
                # label = p/Z
                labels += [log_prediction.exp().cpu()]
            
            labels = torch.cat(labels, dim=0)
            max_pos = labels.argmax(1)
            labels = F.one_hot(max_pos, num_classes=self.output_dim)
            
            return labels
    
    @torch.no_grad()
    def save_pseudo_parameters(self, path, index=0):
        
        # output pseudo centers
        pseudo_path = path + '_pseudo' + str(index) + '.npy'
        pseudo_weight_path = path + '_pseudo_weight' + str(index) + '.npy'
        pseudo_z_mean_path = path + '_pseudo_z_mean' + str(index) + '.npy'
        pseudo_z_logvar_path = path + '_pseudo_z_logvar' + str(index) + '.npy'
        os.makedirs(os.path.dirname(pseudo_path), exist_ok=True)
        
        np.save(pseudo_path, self.pseudo_means(self.idle_input).cpu().data.numpy())
        np.save(pseudo_weight_path, self.pseudo_weights(self.idle_input).cpu().data.numpy())
        
        pseudo_z_mean, pseudo_z_logvar = self.get_pseudo_z()
        np.save(pseudo_z_mean_path, pseudo_z_mean.cpu().data.numpy())
        np.save(pseudo_z_logvar_path, pseudo_z_logvar.cpu().data.numpy())
        
    @torch.no_grad()
    def save_traj_results(self, inputs, batch_size, path, SaveTrajResults, traj_index=0, index=1):
        all_prediction=[] 
        all_z_sample=[] 
        all_z_mean=[] 
        
        for i in range(0, len(inputs), batch_size):
            
            batch_inputs = inputs[i:i+batch_size].to(self.device)
        
            # pass through VAE
            z_mean, z_logvar = self.encode(batch_inputs)
            z_sample = self.reparameterize(z_mean, z_logvar)
        
            log_prediction = self.decoder(z_mean)
            
            all_prediction+=[log_prediction.exp().cpu()]
            all_z_sample+=[z_sample.cpu()]
            all_z_mean+=[z_mean.cpu()]
            
        all_prediction = torch.cat(all_prediction, dim=0)
        all_z_sample = torch.cat(all_z_sample, dim=0)
        all_z_mean = torch.cat(all_z_mean, dim=0)
        
        max_pos = all_prediction.argmax(1)
        labels = F.one_hot(max_pos, num_classes=self.output_dim)
        
        # save the fractional population of different states
        population = torch.sum(labels,dim=0).float()/len(inputs)
        
        population_path = path + '_state_population' + str(index) + '.npy'
        os.makedirs(os.path.dirname(population_path), exist_ok=True)
        
        np.save(population_path, population.cpu().data.numpy())
        
        if SaveTrajResults:
        
            label_path = path + '_traj%d_labels'%(traj_index) + str(index) + '.npy'
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            
            np.save(label_path, labels.cpu().data.numpy())
            
            prediction_path = path + '_traj%d_data_prediction'%(traj_index) + str(index) + '.npy'
            representation_path = path + '_traj%d_representation'%(traj_index) + str(index) + '.npy'
            mean_representation_path = path + '_traj%d_mean_representation'%(traj_index) + str(index) + '.npy'
            
            os.makedirs(os.path.dirname(mean_representation_path), exist_ok=True)
            
            np.save(prediction_path, all_prediction.cpu().data.numpy())
            np.save(representation_path, all_z_sample.cpu().data.numpy())
            np.save(mean_representation_path, all_z_mean.cpu().data.numpy())
            
            self.save_pseudo_parameters(path, index)

                
            
            
        
