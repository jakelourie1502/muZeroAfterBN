import numpy as np
from .frozen_lake import gridWorld
from .mcts import MCTS, Node
from model.representation import Representation
from global_settings import gamma
from global_settings import lake_size, goals, lakes, dist, actions_size, max_plays, play_random, lake_coverage
from utils import vector2d_to_tensor_for_model
import torch

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
        for met in ['policy','action','obs','reward']:
            metrics[met] = []
        obs = self.env.reset()
        metrics['obs'].append(obs) #1 x 1 x obs(1) x obs(2)
        
        while True:
            # self.env.render()
            obs = vector2d_to_tensor_for_model(obs) #need to double unsqueeze here to create fictional batch and channel dims
            state = self.representation_model(obs.float()) #state comes out as a (1,36,8,8)
            root_node = Node(parent='null',state=state)
            policy, action = mcts.one_turn(root_node)
            obs, _, reward, done = self.env.step(action)
            self.store_metrics(policy, action, reward, obs,metrics)
            if done == True:
                break #params for ending episode
        
        self.calculate_V_from_rewards(metrics) #using N step returns or whatever to calculate the returns.
        metrics['obs'] = metrics['obs'][:-1] #otherwise we'd have one extra observation.
        return metrics
        
    def store_metrics(self,policy, action, reward,obs, metrics):
        metrics['obs'].append(obs)
        metrics['policy'].append(policy)
        metrics['action'].append(action)
        metrics['reward'].append(reward)
    
    def calculate_V_from_rewards(self,metrics):
        #### We're going to use fully discounted future returns all the way to end of the episode
        rewards_list = metrics['reward'].copy()
        for r in reversed(range(len(rewards_list)-1)):
            rewards_list[r]+= rewards_list[r+1] * self.gamma
        metrics['V'] = rewards_list
