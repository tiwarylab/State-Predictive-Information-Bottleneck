

# SPIB -- State Predictive Information Bottleneck
SPIB is a deep learning-based framework that learns the reaction coordinates from high dimensional molecular simulation trajectories. Please read and cite this manuscript when using SPIB: https://aip.scitation.org/doi/abs/10.1063/5.0038198. Here is an implementation of SPIB in Pytorch. Please check our demonstrative colab [demo](https://github.com/tiwarylab/State-Predictive-Information-Bottleneck/blob/main/SPIB_Demo.ipynb) and [tutorials](https://github.com/tiwarylab/SPIB-plumed-masterclass-2022). A ```spib``` package can also be found at https://pypi.org/project/spib/.


## Data Preparation
Our implementation now only supports the npy files as the input, and also saves all the results into npy files for further anlyses. Users can refer to the data files in the ```examples``` subdirectory.

## Usage

To train and test model, we proposed two ways:

### For preliminary analyses
```
python test_model.py	-dt	# Time delay delta t in terms of # of minimal time resolution of the trajectory data
			-d	# Dimension of RC or bottleneck
			-encoder_type	# Encoder type (Linear or Nonlinear)
			-n1	# Number of nodes in each hidden layer of the encoder
			-n2	# Number of nodes in each hidden layer of the decoder
			-bs # Batch size
			-threshold	# Threshold of the predicted state population change used to measure the convergence of training for each iteration
			-patience	# Number of epochs with the change of the state population smaller than the threshold after which this iteration of training finishes
			-refinements	# Number of refinements
			-lr	# Learning rate of Adam optimizer
			-b	# Hyperparameter beta
			-label	# Path to the initial state labels
			-traj	# Path to the trajectory data
			-w	# Path to the weights of the samples
			-seed	# Random seed
			-UpdateLabel	# Whether to refine the labels during the training process
			-SaveTrajResults	# Whether save trajectory results
```

#### Example

Train and test SPIB on the four-well analytical potential:
```
python test_model.py -dt 50 -d 1 -encoder_type Nonlinear -bs 512 -threshold 0.01 -patience 2 -refinements 8 -lr 0.001 -b 0.01 -seed 0 -label examples/Four_Well_beta3_gamma4_init_label10.npy -traj examples/Four_Well_beta3_gamma4_traj_data.npy
```

### For advanced analyses
```
python test_model_advanced.py	-config	# Input the configuration file 
```

Here, a configuration file in INI format is supported, which allows a more flexible control of the training process. A sample configuration file is shown in the ```examples``` subdirectory. Two advanced features are included: 
* It supports simple grid search to tune the hyper-parameters;
* It also allows multiple trajectories with different weights as the input data; 

#### Example

Train and test SPIB on the four-well analytical potential:
```
python test_model_advanced.py -config examples/sample_config.ini
```
