import numpy as np
import torch

### LAKE and play
lake_size = (8,8)
goals = {(lake_size[0]-1,lake_size[1]-1):1}

#### Probability of being a in a certain spot inversely proportional to how far it is from goal.
dist = np.zeros((lake_size[0]*lake_size[1]+1))
for v in range(len(dist)-2):
    dist[v] = 1/(v+1)
dist = dist/ np.sum(dist)

lakes = []
lake_coverage = 0.1
max_plays = 200
play_random = 0.1
time_limit_mcts = 0.1

#### Representation and Dynamics model
observation_size = (1,8,8)
hidden_rep_size = (18,6,6)
enc_size = (36,6,6)
actions_size = 4

#### Prediction model
hidden_policy_size = 48
hidden_fut_reward_size = 48
batch_size = 1028
workers = 16
gamma = 0.99

#### MCTS params
c1 = 1.25
c2 = 19652
gamma = 0.97

#### Main function
epochs = 10000
replay_buffer_size = 200000

#### Training params
training_params = {'lr': 0.0001,
                 'optimizer' : torch.optim.SGD,
                 'k': 5,
                 'reward_coef':0.1,
                 'future_val_coef': 0.5,
                 'policy_coef': 0.4
                 }