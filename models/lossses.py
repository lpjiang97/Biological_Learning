import torch
import torch.nn as nn
import torch.nn.functional as F


class OnehotLoss(nn.Module):

    def __init__(self, m=6):
        super(OnehotLoss, self).__init__()
        self.m = m

    def forward(self, x, target):
        x = F.tanh(x)
        batch_size = target.shape[0]
        with torch.no_grad():
            label = F.one_hot(target, num_classes=10)
            label[label == 0] = -1
        return torch.pow(torch.abs(x - label), self.m).sum(1).sum(0) / batch_size
