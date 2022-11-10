import torch.nn as nn

class MLP(nn.Module):
    """用于训练二分类问题"""
    def __init__(self, input_features, out_features):
        super(MLP, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_features, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(10, out_features),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        x = self.hidden(x)
        y = self.classifier(x)
        
        return y