import numpy as np

### LAKE and play
lake_size = (12,12)
goals = {(lake_size[0]-1,lake_size[1]-1):1}
dist = np.zeros((lake_size[0]*lake_size[1]+1))
dist[0]=1
lakes = []
lake_coverage = 0.2
max_plays = 300
play_random = 0.1
time_limit_mcts = 0.1

#### Representation and Dynamics model
observation_size = (1,12,12)
hidden_rep_size = (18,8,8)
enc_size = (36,8,8)
actions_size = 4

#### Prediction model
hidden_policy_size = 48
hidden_fut_reward_size = 48
batch_size = 50
workers = 2
gamma = 0.99

#### MCTS params
c1 = 1.25
c2 = 19652
gamma = 0.97

#### Main function
epochs = 10
replay_buffer_size = 500000
