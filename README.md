

# SPIB -- State Predictive Information Bottleneck
SPIB is a deep learning-based framework that learns the reaction coordinates from high dimensional molecular simulation trajectories. Here is an implitation of SPIB in Pytorch.


## Data Preparation
Our implementation now only supports the npy files as the input, and also saves all the results into npy files for further anlyses. Users can refer to the data files in the ```examples``` subdirectory.

## Usage

To train and test model:
```
python test_model.py	-dt	# Time delay delta t in terms of # of minimal time resolution of the trajectory data
						-d	# Dimension of RC or bottleneck
						-K	# Number of pseudo inputs
						-n1	# Number of nodes in each hidden layer of the encoder
						-n2	# Number of nodes in each hidden layer of the decoder
						-epochs
						-m	# Refinement interval in terms of # of training steps
						-bs # Batch size
						-lr	# Learning rate of Adam optimizer
						-b	# Hyperparameter beta
						-label	# Path to the initial state labels
						-traj	# Path to the trajectory data
						-biased	# Path to the weights of the samples if bised MD is used
						-seed	# Random seed
```
### Example

Train and test SPIB on the four-well analytical potential:
```
python test_model.py -dt 50 -d 1 -K 10 -epochs 10 -bs 512 -m 1000 -lr 0.001 -b 0.01 -seed 0 -label examples/Four_Well_beta3_gamma4_init_label10.npy -traj examples/Four_Well_beta3_gamma4_traj_data.npy
```

