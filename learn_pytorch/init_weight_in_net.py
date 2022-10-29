# 神经网络中进行权重初始化
from cv2 import mean
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv2 = nn.Conv2d(3, 16, 3)
        self.hidden = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU()
        )
        self.output = nn.Linear(50, 10)
        
    def forward(self, x):
        x = self.conv2(x)
        x = self.hidden(x)
        y = self.output(x)
        
        return y


def init_weight(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, mean=0, std=1)  # 若不设置偏置，则为随机值
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight, a=-0.1, b=0.1)  # (-0.1,0.1)的均匀分布
        nn.init.constant_(m.bias, val=0.01)
        
simple_net = SimpleNet()
torch.manual_seed(1)
simple_net.apply(init_weight)
print(simple_net)