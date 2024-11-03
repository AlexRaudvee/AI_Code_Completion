import torch 

# set device
"""
set 'cpu' if you are using not MacOS M1 or M2 chip
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')