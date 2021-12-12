import sys
sys.path.append("..")
import numpy as np
from game_play.frozen_lake import gridWorld
from game_play.play_episode import Episode
import global_settings
from model.representation import Representation
from model.dynamic import Dynamic
from model.prediction import Prediction
from training_and_data.replay_buffer import Replay_Buffer
from global_settings import epochs

from game_play.mcts import MCTS, Node
from global_settings import gamma
from global_settings import lake_size, goals, lakes, dist, actions_size, max_plays, play_random, lake_coverage
from utils import vector2d_to_tensor_for_model
import torch

###### PARAMETERS
epochs = 1
worker = 4
max_plays = 8
dist = np.zeros((lake_size[0] * lake_size[1])+1)
dist[-3] = 1

class Episode:
    def __init__(self,models):
        self.models = models
        self.env=gridWorld(lake_size,lakes,goals, n_actions = actions_size, max_steps = max_plays, dist = dist, seed = None, rnd=play_random)
        
        self.env.generate_random_lakes(lake_coverage)
        self.representation_model, self.dynamics_model, self.predictions_model = self.models
        self.gamma = gamma

    def play_episode(self):
        mcts = MCTS(self)
        metrics = {}
        for met in ['policy','action','obs','reward','search_value','done']:
            metrics[met] = []
        obs = self.env.reset()
        metrics['obs'].append(obs) #1 x 1 x obs(1) x obs(2)
        
        while True:
            # self.env.render()
            obs = vector2d_to_tensor_for_model(obs) #need to double unsqueeze here to create fictional batch and channel dims
            state = self.representation_model(obs.float()) #state comes out as a (1,36,8,8)
            root_node = Node(parent='null',state=state)
            policy, action, search_value = mcts.one_turn(root_node)
            obs, _, reward, done = self.env.step(action)
            print(obs)
            self.store_metrics(policy, action, reward, obs,metrics,done, search_value)
            if done == True:
                break #params for ending episode
        
        self.calculate_V_from_rewards(metrics) #using N step returns or whatever to calculate the returns.
        metrics['obs'] = metrics['obs'][:-1] #otherwise we'd have one extra observation.
        return metrics
        
    def store_metrics(self,policy, action, reward,obs, metrics,done, search_value):
        metrics['obs'].append(obs)
        metrics['policy'].append(policy)
        metrics['action'].append(action)
        metrics['reward'].append(reward)
        metrics['search_value'].append(search_value)
        metrics['done'].append(done)

    def calculate_V_from_rewards(self,metrics):
        #### We're going to use fully discounted future returns all the way to end of the episode
        rewards_list = metrics['reward'].copy()
        for r in range(len(rewards_list)):
            rewards_list[r] *= self.gamma
        for r in reversed(range(len(rewards_list)-1)):
            rewards_list[r] += rewards_list[r+1] * self.gamma #THIS GETS VALUE
        
        
        metrics['V'] = rewards_list

### LOAD IN PARAMETERS
batch_size = 20
workers = 2
gamma = global_settings.gamma
max_plays = 20

### INITIALISE MODELS
representation_model = Representation()
dynamics_model = Dynamic()
predictions_model = Prediction()
replay_buffer = Replay_Buffer()
models = (representation_model, dynamics_model, predictions_model)

for e in range(epochs):
    #### SELF-PLAY
    for m in range(len(models)): 
        models[m].eval()
    for i in range(workers):
        ep = Episode(models) 
        replay_buffer.add_ep_log(ep.play_episode())
    replay_buffer.purge() #keep replay buffer at reasonable size.
    print(replay_buffer.reward_logs)
    print(replay_buffer.fut_val_logs)
