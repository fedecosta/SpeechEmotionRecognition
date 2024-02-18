import torch
from torch import nn
from focal_loss.focal_loss import FocalLoss


class FocalLossCriterion(nn.Module):

    def __init__(self, gamma = 0.7, weights = None):

        super().__init__()

        self.gamma = gamma
        self.weights = weights # The weights parameter is similar to the alpha value mentioned in the paper
        
        self.init_layers()

    
    def init_layers(self):

        self.softmax_layer = torch.nn.Softmax(dim=-1)
        if self.weights is not None:
            self.criterion = FocalLoss(gamma = self.gamma, weights = self.weights)
        else:
            self.criterion = FocalLoss(gamma = self.gamma)


    def forward(self, logits, target):

        return self.criterion(self.softmax_layer(logits), target)