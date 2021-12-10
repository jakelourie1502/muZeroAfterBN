import torch
import numpy as np

def vector2d_to_tensor_for_model(state):
    return torch.unsqueeze(torch.unsqueeze(torch.tensor(state),0),0)

