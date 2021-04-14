"""
SPIB: A deep learning-based framework to learn RCs 
from MD trajectories. Code maintained by Dedi.

Read and cite the following when using this method:
https://arxiv.org/abs/2011.10127
"""
import numpy as np
import torch
import os
import sys
import configparser
import json

import SPIB
import SPIB_training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
default_device = torch.device("cpu")


def test_model_advanced():
    # Settings
    # ------------------------------------------------------------------------------
    # By default, we save all the results in subdirectories of the following path.
    base_path = "SPIB"
    

    # If there is a configuration file, import the configuration file
    # Otherwise, an error will be reported
    if '-config' in sys.argv:
        config = configparser.ConfigParser(allow_no_value=True)

        config.read(sys.argv[sys.argv.index('-config') + 1])

        # Model parameters
        # Time delay delta t in terms of # of minimal time resolution of the trajectory data
        dt_list = json.loads(config.get("Model Parameters","dt"))
        
        # By default, we use all the all the data to train and test our model
        t0 = 0 
        
        # Dimension of RC or bottleneck
        RC_dim_list = json.loads(config.get("Model Parameters","d"))

        # Number of pseudo inputs
        pseudo_dim_list = json.loads(config.get("Model Parameters","K"))
        
        # Number of nodes in each hidden layer of the encoder
        neuron_num1_list = json.loads(config.get("Model Parameters","neuron_num1"))
        # Number of nodes in each hidden layer of the encoder
        neuron_num2_list = json.loads(config.get("Model Parameters","neuron_num2"))
        
        
        # Training parameters
        training_epochs = int(config.get("Training Parameters","epochs"))
        
        batch_size = int(config.get("Training Parameters","batch_size"))

        # Refinement interval in terms of # of training steps
        refinement_interval = int(config.get("Training Parameters","m"))

        # Minimum refinements
        min_refinements = int(config.get("Training Parameters","min_refinements"))
            
        # By default, we save the model every 10000 steps
        log_interval = int(config.get("Training Parameters","log_interval"))
        
        # learning rate of Adam optimizer
        learning_rate_list = json.loads(config.get("Training Parameters","learning_rate"))
        
        # Hyper-parameter beta
        beta_list = json.loads(config.get("Training Parameters","beta"))
        
        # Import data

        # Path to the trajectory data
        traj_data_path = config.get("Data","traj_data")
        traj_data_path = traj_data_path.replace('[','').replace(']','')
        traj_data_path = traj_data_path.split(',')

        # Load the data
        traj_data_list = [torch.from_numpy(np.load(file_path)).float().to(default_device) for file_path in traj_data_path]
        
        # Path to the initial state labels
        initial_labels_path = config.get("Data","initial_labels")
        initial_labels_path = initial_labels_path.replace('[','').replace(']','')
        initial_labels_path = initial_labels_path.split(',')
        
        traj_labels_list = [torch.from_numpy(np.load(file_path)).float().to(default_device) for file_path in initial_labels_path]
        
        output_dim = traj_labels_list[0].shape[1]
        
        assert len(traj_data_list)==len(traj_labels_list)

        # Path to the weights of the samples
        traj_weights_path = config.get("Data","traj_weights")
        if traj_weights_path == None:
            traj_weights_list = None
            IB_path = os.path.join(base_path, "Unweighted")
        else:
            traj_weights_path = traj_weights_path.replace('[','').replace(']','')
            traj_weights_path = traj_weights_path.split(',')
        
            traj_weights_list = [torch.from_numpy(np.load(file_path)).float().to(default_device) for file_path in traj_weights_path]
            IB_path = os.path.join(base_path, "Weighted")
            assert len(traj_weights_list)==len(traj_labels_list)

        
        # Other controls

        # Random seed
        seed_list = json.loads(config.get("Other Controls","seed"))

        # Whether to refine the labels during the training process
        UpdateLabel = bool(config.get("Other Controls","UpdateLabel"))
        
        # Whether save trajectory results
        SaveTrajResults = bool(config.get("Other Controls","SaveTrajResults"))

    else:
        print("Pleast input the config file!")
        return

    
    
    # Train and Test our model
    # ------------------------------------------------------------------------------

    final_result_path = IB_path + '_result.dat'
    os.makedirs(os.path.dirname(final_result_path), exist_ok=True)
    print("Final Result", file=open(final_result_path, 'w'))
    
    for seed in seed_list:
        np.random.seed(seed)
        torch.manual_seed(seed)

        for dt in dt_list:
            data_init_list = [] 
            if traj_weights == None:
                for i in range(len(traj_data_list)):
                    data_init_list+=[SPIB_training.data_init(t0, dt, traj_data, traj_labels, None)]
                train_data_weights = None
                test_data_weights = None
            else:
                for i in range(len(traj_data_list)):
                    data_init_list+=[SPIB_training.data_init(t0, dt, traj_data, traj_labels, None)]

                train_data_weights = torch.cat([data_init_list[i][4] for i in range(len(traj_data_list))], dim=0)
                test_data_weights = torch.cat([data_init_list[i][8] for i in range(len(traj_data_list))], dim=0)

            data_shape = data_init_list[0][0]
            train_past_data = torch.cat([data_init_list[i][1] for i in range(len(traj_data_list))], dim=0)
            train_future_data = torch.cat([data_init_list[i][2] for i in range(len(traj_data_list))], dim=0)
            train_data_labels = torch.cat([data_init_list[i][3] for i in range(len(traj_data_list))], dim=0)

            test_past_data = torch.cat([data_init_list[i][5] for i in range(len(traj_data_list))], dim=0)
            test_future_data = torch.cat([data_init_list[i][6] for i in range(len(traj_data_list))], dim=0)
            test_data_labels = torch.cat([data_init_list[i][7] for i in range(len(traj_data_list))], dim=0)

            for RC_dim in RC_dim_list:
                for pseudo_dim in pseudo_dim_list:
                    for neuron_num1 in neuron_num1_list:
                        for neuron_num2 in neuron_num2_list:
                            for beta in beta_list:
                                for learning_rate in learning_rate_list:

                                    output_path = IB_path + "_d=%d_K=%d_t=%d_b=%.4f_learn=%f" \
                                        % (RC_dim, pseudo_dim, dt, beta, learning_rate)

                                    IB = SPIB.SPIB(RC_dim, pseudo_dim, output_dim, data_shape, device, \
                                                UpdateLabel, neuron_num1, neuron_num2)
                                    
                                    IB.to(device)
                                    
                                    train_result = False
                                    
                                    optimizer = torch.optim.Adam(IB.parameters(), lr=learning_rate)

                                    train_result = SPIB_training.train(IB, beta, train_past_data, train_future_data, \
                                                                    train_data_labels, train_data_weights, test_past_data, test_future_data, \
                                                                        test_data_labels, test_data_weights, optimizer, \
                                                                            training_epochs, refinement_interval, batch_size, min_refinements, output_path, \
                                                                                log_interval, device, seed)
                                    
                                    if train_result:
                                        return
                                    
                                    SPIB_training.output_final_result(IB, device, train_past_data, train_future_data, train_data_labels, train_data_weights, \
                                                                    test_past_data, test_future_data, test_data_labels, test_data_weights, batch_size, \
                                                                        output_path, final_result_path, dt, beta, learning_rate, seed)

                                    for i in range(len(traj_data_list)):
                                        IB.save_traj_results(traj_data_list[i], batch_size, output_path, SaveTrajResults, i, seed)
                                    
                                    IB.save_pseudo_parameters(output_path, seed)


if __name__ == '__main__':
    
    test_model()
    

    
    
