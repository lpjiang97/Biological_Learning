from audioop import bias
import torch
import torch.nn as nn
import torch.nn.functional as F


class BPNet(nn.Module):
    def __init__(self, input_dim, input_channel, hidden_dim, output_dim):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(input_dim * input_channel, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim) 

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class BioNet(nn.Module):

    def __init__(self, bio_weights, output_dim):
        super(BioNet, self).__init__()
        self.bio_layer = nn.Linear(bio_weights.shape[1], bio_weights.shape[0], bias=False)
        self.fc = nn.Linear(bio_weights.shape[0], output_dim)
        self.bio_layer.weight.data = bio_weights

    def forward(self, x):
        return self.fc(F.relu(self.bio_layer(x)))
