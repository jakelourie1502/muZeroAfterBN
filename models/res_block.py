import torch
import numpy as np 
from global_settings import state_channels, res_block_high_channels, res_block_kernel_size
class resBlock(torch.nn.Module):
    """
    Input: 
      returns a view with the same dims

    """
    def __init__(self,downsample=False):
        super().__init__()
        self.downsample=downsample
        if downsample:
            self.identity_stride = 2
        else:
            self.identity_stride = 1
        self.conv1 = torch.nn.Conv2d(in_channels = state_channels,out_channels = res_block_high_channels,kernel_size = res_block_kernel_size,stride = 1,padding = res_block_kernel_size // 2)
        self.bn1 = torch.nn.BatchNorm2d(res_block_high_channels)
        self.convIdentity = torch.nn.Conv2d(in_channels = res_block_high_channels, out_channels = state_channels, kernel_size = 1, stride = self.identity_stride, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(state_channels)
        self.outer_identity = torch.nn.Conv2d(in_channels = state_channels, out_channels = state_channels, kernel_size = 1, stride = self.identity_stride, padding=0)
        self.bn3 = torch.nn.BatchNorm2d(state_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.convIdentity(x)
        x = self.bn2(x)
        if self.downsample:
            identity = self.outer_identity(identity)
            identity = self.bn3(identity)
        x += identity
        x = self.relu(x)
        return x

class ResBlockLinear(torch.nn.Module):
    """"

    """
    def __init__(self,dims):
        super().__init__()
        self.lin1 = torch.nn.Linear(dims[0], dims[1])
        self.lin2 = torch.nn.Linear(dims[1], dims[0])
        self.relu = torch.nn.ReLU()
    
    def forward(self,x):
        identity = x
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x += identity
        x = self.relu(x)
        return x