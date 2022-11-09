import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class BiMLP(nn.Module):
    """用于训练二分类问题"""
    def __init__(self, input_features):
        super(BiMLP, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_features, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(10, 2),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        x = self.hidden(x)
        y = self.classifier(x)
        
        return y