from audioop import bias
from colorsys import TWO_THIRD
from turtle import hideturtle
import torch
import torch.nn as nn
import torch.nn.functional as F


class BPNet(nn.Module):
    def __init__(self, input_dim, input_channel, hidden_dim, output_dim, two_layer=False):
        super(BPNet, self).__init__()
        layers = [
            nn.Linear(input_dim * input_channel, hidden_dim),
            nn.ReLU(),
        ]
        if two_layer:
            layers += [nn.Linear(hidden_dim, 100),
                       nn.ReLU(),
                       nn.Linear(100, output_dim)]
        else:
            layers.append(nn.Linear(hidden_dim, output_dim)) 

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class BioNet(nn.Module):

    def __init__(self, first_bio_weights, output_dim, n=1, two_layer=False, second_bio_weights=None):
        super(BioNet, self).__init__()
        self.first_bio_layer = nn.Linear(first_bio_weights.shape[1], first_bio_weights.shape[0], bias=False)
        self.sec_bio_layer = None
        self.fc = nn.Linear(100, output_dim) if two_layer else nn.Linear(first_bio_weights.shape[0], output_dim)
        self.first_bio_layer.weight.data = first_bio_weights
        self.n = n
        self.two_layer = two_layer
        if two_layer:
            assert second_bio_weights is not None
            self.sec_bio_layer = nn.Linear(first_bio_weights.shape[0], 100, bias=False)
            self.sec_bio_layer.weight.data = second_bio_weights

    def forward(self, x):
        x = torch.pow(F.relu(self.first_bio_layer(x)), self.n)
        if self.two_layer:
            x = self.fc(torch.pow(F.relu(self.sec_bio_layer(x)), self.n)) 
        else:
            x = self.fc(x) 
        return x
