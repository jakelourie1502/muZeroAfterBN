import time
import numpy as np
from global_settings import c1, c2 , gamma, actions_size, time_limit_mcts
from utils import vector2d_to_tensor_for_model
import torch

class Node:
    def __init__(self,parent,state, r = 0, p = 0):
        self.Q_sum = 0
        self.Q = 0
        self.N = 0
        self.P = p
        self.R = r
        self.fut_v = 0
        self.parent = parent
        self.children = []
        self.state = state
        
class MCTS:
    def __init__(self, ep):
        self.ep = ep
        self.c1, self.c2, self.gamma = c1,c2, gamma
        self.num_actions = actions_size
    
    def one_turn(self,root_node,time_limit=time_limit_mcts):
        tn = time.time()
        self.nodes = []
        self.root_node = root_node 
        idx = 0 
        while time.time() < tn + time_limit:
            idx+=1
            self.mcts_go(root_node)
        policy, chosen_action = self.randomly_sample_action(self.root_node)
        print(idx)
        return policy, chosen_action
        
    def mcts_go(self,node):
        if len(node.children) == 0:
            self.expand(node)
        else:
            best_ucb_child = self.pick_child(node)
            self.mcts_go(best_ucb_child)
                
    def expand(self,node):
        """You've reached a terminal node. Backpropogate the rewards and expand the node."""
        prob_action,V = self.ep.predictions_model(node.state) #this will come out as a 1 x 4 and 1x1
        prob_action, V = prob_action[0], V[0][0]
        self.back_prop_rewards(node, V)
        ## Add a child node for each action of this node.
        for edge in range(self.num_actions):
            state, r = self.ep.dynamics_model(node.state,edge) #
            r = r[0][0]
            new_node = Node(parent=node, state=state, r=r, p=prob_action[edge])
            node.children.append(new_node)
            self.nodes.append(new_node)
        
    def pick_child(self,node):
        return node.children[np.argmax([self.UCB_calc(x).detach().numpy() for x in node.children])]
    
    def UCB_calc(self,node):        
        policy_and_novelty_coef = node.P * np.sqrt(node.parent.N) / (1+node.N)
        muZeroModerator = self.c1 + np.log((node.parent.N + self.c2 + self.c1+1)/self.c2)
        return node.Q + policy_and_novelty_coef * muZeroModerator
    
    def back_prop_rewards(self, node, V):
        """just send those rewards up the chain"""
        node.Q_sum += V + node.R
        node.N +=1
        node.Q = node.Q_sum / node.N
        if node != self.root_node:
            V = node.Q * self.gamma
            self.back_prop_rewards(node.parent, V)

    def randomly_sample_action(self,root_node):
        policy = np.array([x.N + + 1e-5 for x in root_node.children ])
        policy = policy / np.sum(policy)
        return policy, np.random.choice(list(range(actions_size)), p=policy)

    def pick_best_action(self, root_node):
        policy = np.array([x.N for x in root_node.children])
        return policy, np.argmax(policy)