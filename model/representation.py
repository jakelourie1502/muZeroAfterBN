import torch
from global_settings import enc_size, hidden_rep_size, observation_size
class Representation(torch.nn.Module):
    """
    Input: 
      Takes in an observation, and returns a state.
    Notes:
     
    Outputs: 
      a state representation

    """
    def __init__(self):
        self.enc_size = enc_size
        self.hidden_size = hidden_rep_size
        self.obs_size = observation_size
        super().__init__()
        self.conv1 = torch.nn.Conv2d(self.obs_size[0],self.hidden_size[0],kernel_size= 3, padding=0) 
        self.conv2 = torch.nn.Conv2d(self.hidden_size[0],self.hidden_size[0],kernel_size= 3, padding=1) 
        self.conv3 = torch.nn.Conv2d(self.hidden_size[0],self.hidden_size[0],kernel_size= 3, padding=1) 
        self.conv4 = torch.nn.Conv2d(self.hidden_size[0], self.enc_size[0], kernel_size= 3, padding=1)        
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.tanh(x)
        return x

