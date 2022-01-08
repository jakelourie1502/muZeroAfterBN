import sys
sys.path.append(".")
import torch
import torch.nn.functional as TF 
from global_settings import env_size, state_channels, global_hidden_layer, policy_output_supp, policy_hidden_dim, value_hidden_dim, value_output_supp
from models.res_block import ResBlockLinear
class Prediction(torch.nn.Module):
    """
    Input: 
      Takes in an observation, and returns a state.
    Notes:
     
    Outputs: 
      a state representation

    """
    def __init__(self):
        super().__init__()
        
        self.FC1v = torch.nn.Linear(state_channels * env_size[0]*env_size[1], global_hidden_layer)        
        self.bn1v = torch.nn.BatchNorm1d(global_hidden_layer)
        self.FChidV = torch.nn.Linear(global_hidden_layer, value_hidden_dim)
        self.bn2v = torch.nn.BatchNorm1d(value_hidden_dim)
        self.FCoutV = torch.nn.Linear(value_hidden_dim, value_output_supp)
        
        self.FC1p = torch.nn.Linear(state_channels * env_size[0]*env_size[1], global_hidden_layer)        
        self.bn1p = torch.nn.BatchNorm1d(global_hidden_layer)
        self.FChidp = torch.nn.Linear(global_hidden_layer, policy_hidden_dim)
        self.bn2p = torch.nn.BatchNorm1d(value_hidden_dim)
        self.FCoutP = torch.nn.Linear(policy_hidden_dim, policy_output_supp)
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
        self.sm = torch.nn.Softmax(dim=1)
    
    def forward(self,state):
        x = torch.flatten(state,start_dim=1)
        v = x 
        p = x
        

        ##value
        v = self.FC1v(v)
        v = self.bn1v(v)
        v = self.relu(v)
        v = self.FChidV(v)
        v = self.bn2v(v)
        v = self.relu(v)
        v = self.FCoutV(v)
        v = self.sig(v)

        ##policy
        p = self.FC1p(p)
        p = self.bn1p(p)
        p = self.relu(p)
        p = self.FChidp(p)
        p = self.bn2p(p)
        p = self.relu(p)
        p = self.FCoutP(p)
        p = self.sm(p)
        return p,v