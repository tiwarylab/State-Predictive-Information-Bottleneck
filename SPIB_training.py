"""
SPIB: A deep learning-based framework to learn RCs 
from MD trajectories. Code maintained by Dedi.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/5.0038198
"""
import torch
import numpy as np
import time
import os

# Data Processing
# ------------------------------------------------------------------------------

def data_init(t0, dt, traj_data, traj_label, traj_weights):
    assert len(traj_data)==len(traj_label)
    
    # skip the first t0 data
    past_data = traj_data[t0:(len(traj_data)-dt)]
    future_data = traj_data[(t0+dt):len(traj_data)]
    label = traj_label[(t0+dt):len(traj_data)]
    
    # data shape
    data_shape = past_data.shape[1:]
    
    n_data = len(past_data)
    
    # 90% random test/train split
    p = np.random.permutation(n_data)
    past_data = past_data[p]
    future_data = future_data[p]
    label = label[p]
    
    past_data_train = past_data[0: (9 * n_data) // 10]
    past_data_test = past_data[(9 * n_data) // 10:]
    
    future_data_train = future_data[0: (9 * n_data) // 10]
    future_data_test = future_data[(9 * n_data) // 10:]
    
    label_train = label[0: (9 * n_data) // 10]
    label_test = label[(9 * n_data) // 10:]
    
    if traj_weights != None:
        assert len(traj_data)==len(traj_weights)
        weights = traj_weights[t0:(len(traj_data)-dt)]
        weights = weights[p]
        weights_train = weights[0: (9 * n_data) // 10]
        weights_test = weights[(9 * n_data) // 10:]
    else:
        weights_train = None
        weights_test = None
    
    return data_shape, past_data_train, future_data_train, label_train, weights_train,\
        past_data_test, future_data_test, label_test, weights_test

# Loss function
# ------------------------------------------------------------------------------

def calculate_loss(IB, data_inputs, data_targets, data_weights, beta=1.0):
    
    # pass through VAE
    outputs, z_sample, z_mean, z_logvar = IB.forward(data_inputs)
    
    # KL Divergence
    log_p = IB.log_p(z_sample)
    log_q = -0.5 * torch.sum(z_logvar + torch.pow(z_sample-z_mean, 2)
                             /torch.exp(z_logvar), dim=1)
    
    if data_weights == None:
        # Reconstruction loss is cross-entropy
        reconstruction_error = torch.mean(torch.sum(-data_targets*outputs, dim=1))
        
        # KL Divergence
        kl_loss = torch.mean(log_q-log_p)
        
    else:
        # Reconstruction loss is cross-entropy
        # reweighed
        reconstruction_error = torch.mean(data_weights*torch.sum(-data_targets*outputs, dim=1))
        
        # KL Divergence
        kl_loss = torch.mean(data_weights*(log_q-log_p))
        
    
    loss = reconstruction_error + beta*kl_loss

    return loss, reconstruction_error.float(), kl_loss.float()


# Train and test model
# ------------------------------------------------------------------------------

def sample_minibatch(past_data, data_labels, data_weights, indices, device):
    sample_past_data = past_data[indices].to(device)
    sample_data_labels = data_labels[indices].to(device)
    
    if data_weights == None:
        sample_data_weights = None
    else:
        sample_data_weights = data_weights[indices].to(device)
    
    
    return sample_past_data, sample_data_labels, sample_data_weights


def train(IB, beta, train_past_data, train_future_data, init_train_data_labels, train_data_weights, \
          test_past_data, test_future_data, init_test_data_labels, test_data_weights, \
              optimizer, n_epoch, refinement_interval, batch_size, min_refinements, output_path, log_interval, device, index):
    IB.train()
    
    step = 0
    start = time.time()
    log_path = output_path + '_train.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    IB_path = output_path + "cpt" + str(index) + "/IB"
    os.makedirs(os.path.dirname(IB_path), exist_ok=True)
    
    train_data_labels = init_train_data_labels
    test_data_labels = init_test_data_labels

    updata_times = 0
    epoch = 0

    # Stop only if epoch >= n_epoch and updata_times >= min_refinements
    while epoch < n_epoch or updata_times < min_refinements:
        
        train_permutation = torch.randperm(len(train_past_data))
        test_permutation = torch.randperm(len(test_past_data))
        
        
        for i in range(0, len(train_past_data), batch_size):
            step += 1
            
            if i+batch_size>len(train_past_data):
                break
            
            train_indices = train_permutation[i:i+batch_size]
            
            batch_inputs, batch_outputs, batch_weights = sample_minibatch(train_past_data, train_data_labels, \
                                                                       train_data_weights, train_indices, device)
                    
            loss, reconstruction_error, kl_loss= calculate_loss(IB, batch_inputs, \
                                                                batch_outputs, batch_weights, beta)
            
            # Stop if NaN is obtained
            if(torch.isnan(loss).any()):
                return True
    
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            if step % 500 == 0:
                with torch.no_grad():
                    
                    batch_inputs, batch_outputs, batch_weights = sample_minibatch(train_past_data, train_data_labels, \
                                                                               train_data_weights, train_indices, device)
                            
                    loss, reconstruction_error, kl_loss= calculate_loss(IB, batch_inputs, \
                                                                        batch_outputs, batch_weights, beta)
                    train_time = time.time() - start
            
                    print(
                        "Iteration %i:\tTime %f s\nLoss (train) %f\tKL loss (train): %f\n"
                        "Reconstruction loss (train) %f" % (
                            step, train_time, loss, kl_loss, reconstruction_error))
                    print(
                       "Iteration %i:\tTime %f s\nLoss (train) %f\tKL loss (train): %f\n"
                        "Reconstruction loss (train) %f" % (
                            step, train_time, loss, kl_loss, reconstruction_error), file=open(log_path, 'a'))
                    j=i%len(test_permutation)
                    
                    
                    
                    test_indices = test_permutation[j:j+batch_size]
                    
                    batch_inputs, batch_outputs, batch_weights = sample_minibatch(test_past_data, test_data_labels, \
                                                                               test_data_weights, test_indices, device)
                    
                    loss, reconstruction_error, kl_loss = calculate_loss(IB, batch_inputs, \
                                                                         batch_outputs, batch_weights, beta)

                    train_time = time.time() - start
                    print(
                       "Loss (test) %f\tKL loss (test): %f\n"
                       "Reconstruction loss (test) %f" % (
                           loss, kl_loss, reconstruction_error))
                    print(
                       "Loss (test) %f\tKL loss (test): %f\n"
                       "Reconstruction loss (test) %f" % (
                           loss, kl_loss, reconstruction_error), file=open(log_path, 'a'))
                    
            # label update
            if IB.UpdateLabel and step % refinement_interval == 0:
                train_data_labels = IB.update_labels(train_future_data, batch_size)
                test_data_labels = IB.update_labels(test_future_data, batch_size)

                updata_times+=1
                print("Update %d\n"%updata_times)
        
            if step % log_interval == 0:
                # save model
                torch.save({'step': step,
                            'state_dict': IB.state_dict()},
                           IB_path+ '_%d_cpt.pt'%step)
                torch.save({'optimizer': optimizer.state_dict()},
                           IB_path+ '_%d_optim_cpt.pt'%step) 

        epoch+=1
        
                
    # output the saving path
    total_training_time = time.time() - start
    print("Total training time: %f" % total_training_time)
    print("Total training time: %f" % total_training_time, file=open(log_path, 'a'))
    # save model
    torch.save({'step': step,
                'state_dict': IB.state_dict()},
               IB_path+ '_%d_cpt.pt'%step)
    torch.save({'optimizer': optimizer.state_dict()},
               IB_path+ '_%d_optim_cpt.pt'%step)
    
    return False

@torch.no_grad()
def output_final_result(IB, device, train_past_data, train_future_data, train_data_labels, train_data_weights, \
                        test_past_data, test_future_data, test_data_labels, test_data_weights, batch_size, output_path, \
                            path, dt, beta, learning_rate, index=0):
    
    with torch.no_grad():
        final_result_path = output_path + '_final_result' + str(index) + '.npy'
        os.makedirs(os.path.dirname(final_result_path), exist_ok=True)
        
        # label update
        if IB.UpdateLabel:
            train_data_labels = IB.update_labels(train_future_data, batch_size)
            test_data_labels = IB.update_labels(test_future_data, batch_size)
        
        final_result = []
        # output the result
        
        loss, reconstruction_error, kl_loss= [0 for i in range(3)]
        
        for i in range(0, len(train_past_data), batch_size):
            batch_inputs, batch_outputs, batch_weights = sample_minibatch(train_past_data, train_data_labels, train_data_weights, \
                                                                       range(i,min(i+batch_size,len(train_past_data))), IB.device)
            loss1, reconstruction_error1, kl_loss1 = calculate_loss(IB, batch_inputs, batch_outputs, \
                                                                    batch_weights, beta)
            loss += loss1*len(batch_inputs)
            reconstruction_error += reconstruction_error1*len(batch_inputs)
            kl_loss += kl_loss1*len(batch_inputs)
            
        
        # output the result
        loss/=len(train_past_data)
        reconstruction_error/=len(train_past_data)
        kl_loss/=len(train_past_data)
                
        final_result += [loss.data.cpu().numpy(), reconstruction_error.cpu().data.numpy(), kl_loss.cpu().data.numpy()]
        print(
            "Final: %d\nLoss (train) %f\tKL loss (train): %f\n"
                    "Reconstruction loss (train) %f" % (
                index, loss, kl_loss, reconstruction_error))
        print(
            "Final: %d\nLoss (train) %f\tKL loss (train): %f\n"
                    "Reconstruction loss (train) %f" % (
                index, loss, kl_loss, reconstruction_error),
            file=open(path, 'a'))
    
        loss, reconstruction_error, kl_loss = [0 for i in range(3)]
        
        for i in range(0, len(test_past_data), batch_size):
            batch_inputs, batch_outputs, batch_weights = sample_minibatch(test_past_data, test_data_labels, test_data_weights, \
                                                                                         range(i,min(i+batch_size,len(test_past_data))), IB.device)
            loss1, reconstruction_error1, kl_loss1 = calculate_loss(IB, batch_inputs, batch_outputs, \
                                                                   batch_weights, beta)
            loss += loss1*len(batch_inputs)
            reconstruction_error += reconstruction_error1*len(batch_inputs)
            kl_loss += kl_loss1*len(batch_inputs)
            
        
        # output the result
        loss/=len(test_past_data)
        reconstruction_error/=len(test_past_data)
        kl_loss/=len(test_past_data)
        
        final_result += [loss.cpu().data.numpy(), reconstruction_error.cpu().data.numpy(), kl_loss.cpu().data.numpy()]
        print(
            "Loss (test) %f\tKL loss (train): %f\n"
            "Reconstruction loss (test) %f"
            % (loss, kl_loss, reconstruction_error))
        print( 
            "Loss (test) %f\tKL loss (train): %f\n"
            "Reconstruction loss (test) %f"
            % (loss, kl_loss, reconstruction_error), file=open(path, 'a'))
        
        print("dt: %d\t Beta: %f\t Learning_rate: %f" % (
            dt, beta, learning_rate))
        print("dt: %d\t Beta: %f\t Learning_rate: %f" % (
            dt, beta, learning_rate),
              file=open(path, 'a'))    
        
        
        final_result = np.array(final_result)
        np.save(final_result_path, final_result)
