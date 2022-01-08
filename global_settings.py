import numpy as np
import torch

### LAKE and play
env_size = (4,4)

goals = {(0,0):1}
dist = np.zeros((env_size[0]*env_size[1]+1))
dist[0] = 1 #obsolete if using random_start
randomly_create_env = True
lakes = []
lake_coverage = [0.1,0.2]
max_plays = 10
play_random = 0

#### Model parameters
actions_size = 4
encodings_per_action = 8
#Repr
hidden_layer_rep_channels = 8
state_channels = 32
hidden_layer_rep_params = {'kernel_size': 1, 'stride': 1, 'padding': 0}
to_state_params = {'kernel_size': 1, 'stride': 1, 'padding': 0}
res_block_reps = [False] #relates to downsampling
#Res
res_block_high_channels  = 64
res_block_kernel_size = 3

#Dynamic
first_layer_dyn_params = {'kernel_size': 3, 'stride': 1, 'padding': 1}
res_block_dyns = [False] #relates to downsampling

#Prediction
global_hidden_layer = 64
value_hidden_dim = 48
value_output_supp = 1
policy_hidden_dim = 48
policy_output_supp = 4


#### Training parameters

#### mcts functions
c1 = 0.25
c2 = 19652
ucb_noise = [0,0.01]
temperature_init = 1
temperature_changes ={-1: 1, 4000: 0.67, 6000: 0.4}
play_limit_mcts = {-1: 9, 1250:13, 1500: 17, 3000: 65}
manual_over_ride_play_limit = None #only used in final testing
exponent_node_n = 2
ucb_denom_k = 0.25
use_policy = True
dirichlet_alpha = 1


#### Main function
loading_in = True
start_epoch = 370
epochs = 1000000
replay_buffer_size = 50000
gamma = 1

#### Training params
batch_size = 1024
batches_per_train = 1
workers = 16
training_params = {'lr': 0.05,
                'lr_warmup': 750,
                'lr_decay': 0.1,
                'lr_decay_steps':2000,
                 'optimizer' : torch.optim.RMSprop,
                 'k': 4,
                 'policy_coef': 0.33,
                 'policy_ramp_up':100,
                 'entropy_coef': 0,
                 'entropy_first100': 1,
                 'l2': 0.0001,
                 'rho': 0.99 
                 }
epsilon_floor = 0.0
epsilon_ramp_epochs = 100
train_start_batch_multiple = 5
prioritised_replay = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
