import torch
from global_settings import hidden_rep_size, actions_size, enc_size

class Dynamic(torch.nn.Module):

    """
    Input: 
      takes state, adds action and returns new state and rewards
    Notes:
     
    Outputs: 
      a state reprensentation and a reward for that transition.

    """
    def __init__(self):
        self.enc_size = enc_size
        self.hidden_size = hidden_rep_size  
        super().__init__()
        self.conv1 = torch.nn.Conv2d(self.enc_size[0]+1,self.hidden_size[0],kernel_size= 3, padding=1) 
        self.conv2 = torch.nn.Conv2d(self.hidden_size[0],self.hidden_size[0],kernel_size= 3, padding=1) 
        self.conv3 = torch.nn.Conv2d(self.hidden_size[0],self.hidden_size[0],kernel_size= 3, padding=1) 
        self.conv4 = torch.nn.Conv2d(self.hidden_size[0], self.enc_size[0], kernel_size= 3, padding=1)
        self.FC1 = torch.nn.Linear(self.enc_size[0]*self.enc_size[1]*self.enc_size[2],1)
        self.relu = torch.nn.ReLU()

    
    def forward(self, state, action):
        """State is of shape (bs, enc.channels, enc.h, enc.h) """
        
        #### Concats actions to state 
        action_bias_plane = torch.zeros(state.shape[0], 1, self.enc_size[1],self.enc_size[2])
        action_bias_plane += action / actions_size
        
        state_action = torch.cat((state, action_bias_plane),dim=1)

        ##### Goes throughg a hidden state and then to next state, before going into reward prediction.
        hidden_state = self.conv1(state_action)
        hidden_state = self.relu(hidden_state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = self.relu(hidden_state)
        hidden_state = self.conv3(hidden_state)
        hidden_state = self.relu(hidden_state)
        next_state = self.conv4(hidden_state)
        next_state_final = self.relu(next_state)
        next_state = torch.flatten(next_state,start_dim=1)
        reward = self.FC1(next_state)
      
        return next_state_final, reward

