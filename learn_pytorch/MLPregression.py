import torch.nn as nn


class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(in_features=8, out_features=100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU()
        )
        self.regression = nn.Linear(50, 1)  # 回归问题最后层不需要激活函数
        
    def forward(self, x):
        h1 = self.hidden(x)
        output = self.regression(h1)    # output.size: (batch_size, 1) 二维向量
        
        return output[:, 0] # 转变成一维向量 (64,)