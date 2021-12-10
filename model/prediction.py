import torch
from global_settings import actions_size, hidden_policy_size, hidden_fut_reward_size, enc_size
class Prediction(torch.nn.Module):
    """
    Input: 
      Takes in a state (hidden) function)
    Notes:
     
    Outputs: 
      predicted policy - this should be a vector of shape (n_actions, )
      predicted future value of the state - just a value (V)

    """
    def __init__(self):
        self.actions_size = actions_size
        self.hidden_policy_size = hidden_policy_size
        self.hidden_fut_reward_size = hidden_fut_reward_size
        self.enc_size = enc_size
        super().__init__()
        self.FC1_policy = torch.nn.Linear(self.enc_size[0]*self.enc_size[1]*self.enc_size[2], self.hidden_policy_size)
        self.FC2_policy = torch.nn.Linear(self.hidden_policy_size,self.actions_size)
        self.FC1_future_reward = torch.nn.Linear(self.enc_size[0]*self.enc_size[1]*self.enc_size[2], self.hidden_fut_reward_size)
        self.FC2_future_reward = torch.nn.Linear(self.hidden_fut_reward_size,1)
        self.sm = torch.nn.Softmax(dim=-1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, state):
        x = torch.flatten(state, start_dim=1)
        policy = self.FC1_policy(x)
        policy = self.relu(policy)
        policy = self.FC2_policy(policy)
        policy = self.sm(policy)
        future_reward = self.FC1_future_reward(x)
        future_reward = self.relu(future_reward)
        future_reward = self.FC2_future_reward(future_reward)
        return policy, future_reward

