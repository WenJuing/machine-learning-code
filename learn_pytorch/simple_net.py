# 简单神经网络的训练
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(in_features=10, out_features=10), # 输入特征数量由数据集而定
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
        )
        self.regression = nn.Linear(10, 1)  # 回归问题，预测结果为一个值
        
    def forward(self, x):
        x = self.hidden(x)
        y = self.regression(x)
        return y