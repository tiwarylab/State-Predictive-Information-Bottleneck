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

import SPIB
import SPIB_training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
default_device = torch.device("cpu")


def test_model():
    # Settings
    # ------------------------------------------------------------------------------
    # By default, we save all the results in subdirectories of the following path.
    base_path = "SPIB"
    
    # Model parameters
    # Time delay delta t in terms of # of minimal time resolution of the trajectory data
    if '-dt' in sys.argv:
        dt = int(sys.argv[sys.argv.index('-dt') + 1])
    else:
        dt = 10
    
    # By default, we use all the all the data to train and test our model
    t0 = 0 
    
    # Dimension of RC or bottleneck
    if '-d' in sys.argv:
        RC_dim = int(sys.argv[sys.argv.index('-d') + 1])
    else:
        RC_dim = 2

    # Number of pseudo inputs
    if '-K' in sys.argv:
        pseudo_dim = int(sys.argv[sys.argv.index('-K') + 1])
    else:
        pseudo_dim = 10
    
    # Number of nodes in each hidden layer of the encoder
    if '-n1' in sys.argv:
        neuron_num1 = int(sys.argv[sys.argv.index('-n1') + 1])
    else:
        neuron_num1 = 16
    # Number of nodes in each hidden layer of the encoder
    if '-n2' in sys.argv:
        neuron_num2 = int(sys.argv[sys.argv.index('-n2') + 1])
    else:
        neuron_num2 = 16
    
    
    # Training parameters
    if '-epochs' in sys.argv:
        training_epochs = int(sys.argv[sys.argv.index('-epochs') + 1])
    else:
        training_epochs = 3
    
    # Refinement interval in terms of # of training steps
    if '-m' in sys.argv:
        refinement_interval = int(sys.argv[sys.argv.index('-m') + 1])
    else:
        refinement_interval = 2000

    if '-bs' in sys.argv:
        batch_size = int(sys.argv[sys.argv.index('-bs') + 1])
    else:
        batch_size = 2048
        
    # By default, we save the model every 10000 steps
    my_log_interval = 10000 
    
    # learning rate of Adam optimizer
    if '-lr' in sys.argv:
        learning_rate = float(sys.argv[sys.argv.index('-lr') + 1])
    else:
        learning_rate = 1e-3
    
    # Hyper-parameter beta
    if '-b' in sys.argv:
        beta = float(sys.argv[sys.argv.index('-b') + 1])
    else:
        beta = 1e-3
    
    # Import data
    
    # Path to the initial state labels
    if '-label' in sys.argv:
        initial_label = np.load(sys.argv[sys.argv.index('-label') + 1])
    else:
        print("Pleast input the initial state labels!")
        return
    
    traj_labels = torch.from_numpy(initial_label).float().to(default_device)
    output_dim = initial_label.shape[1]
    
    # Path to the trajectory data
    if '-traj' in sys.argv:
        traj_data = np.load(sys.argv[sys.argv.index('-traj') + 1])
    else:
        print("Pleast input the trajectory data!")
        return
    
    traj_data = torch.from_numpy(traj_data).float().to(default_device)
    
    
    # Path to the weights of the samples if bised MD is used
    if '-bias' in sys.argv:
        traj_bias = np.load(sys.argv[sys.argv.index('-bias') + 1])
        traj_bias = torch.from_numpy(traj_bias).float().to(default_device)
        BiasReweighed = True
        IB_path = os.path.join(base_path, "Bias")
    else:
        traj_bias = None
        BiasReweighed = False
        IB_path = os.path.join(base_path, "Unbias")
    
    # Random seed
    if '-seed' in sys.argv:
        index = int(sys.argv[sys.argv.index('-seed') + 1])
        np.random.seed(index)
        torch.manual_seed(index)    
    else:
        index = 0
    
    
    # Other controls
    
    # Whether to refine the labels during the training process
    UpdateLabel = True
    
    # Whether save trajectory results
    SaveTrajResults = True
    
    # Train and Test our model
    # ------------------------------------------------------------------------------
    
    final_result_path = IB_path + '_result.dat'
    os.makedirs(os.path.dirname(final_result_path), exist_ok=True)
    print("Final Result", file=open(final_result_path, 'w'))
    
    data_shape, train_past_data, train_future_data, train_data_labels, train_data_bias, \
        test_past_data, test_future_data, test_data_labels, test_data_bias = \
            SPIB_training.data_init(t0, dt, traj_data, traj_labels, traj_bias)
    
    output_path = IB_path + "_d=%d_K=%d_t=%d_b=%.3f_learn=%f" \
        % (RC_dim, pseudo_dim, dt, beta, learning_rate)

    IB = SPIB.SPIB(RC_dim, pseudo_dim, output_dim, data_shape, device, BiasReweighed, \
                   UpdateLabel, neuron_num1, neuron_num2)
    
    IB.to(device)
    
    train_result = False
    
    optimizer = torch.optim.Adam(IB.parameters(), lr=learning_rate)

    train_result = SPIB_training.train(IB, beta, train_past_data, train_future_data, \
                                       train_data_labels, train_data_bias, test_past_data, test_future_data, \
                                           test_data_labels, test_data_bias, optimizer, \
                                               training_epochs, refinement_interval, batch_size, output_path, \
                                                   my_log_interval, device, index)
    
    if train_result:
        return
    
    SPIB_training.output_final_result(IB, device, train_past_data, train_future_data, train_data_labels, train_data_bias, \
                                      test_past_data, test_future_data, test_data_labels, test_data_bias, batch_size, \
                                          output_path, final_result_path, dt, beta, learning_rate, index)

    IB.save_traj_results(traj_data, batch_size, output_path, SaveTrajResults, index)
    
    IB.save_pseudo_parameters(output_path, index)


if __name__ == '__main__':
    
    test_model()
    

    
    
