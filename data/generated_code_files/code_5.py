import torch 
import sys 
import os

import torch.nn as nn 
from config_RNN import device
# Hyperparameters 
sequence_length = 28 

# create RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward prop 
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)#<PREDICTION>

        out = self.fc(out)
        return out

# create CNN
class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(hidden_size*sequence_length, num_classes)
        )

    def forward(self, x):
        out = self.cnn(x)#</PREDICTION>
        
        return out
