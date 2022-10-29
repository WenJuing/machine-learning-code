# 定义优化器
import torch.nn as nn
from torch.optim import Adam


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__(self)()
        self.hidden = nn.Sequential(nn.Linear(784, 10), nn.ReLU())
        self.regression = nn.Linear(10, 1)
        
    def forward(self, x):
        x = self.hidden(x)
        output = self.regression(x)
        return output
    
simple_net = SimpleNet()
# 定义优化器
optimizer = Adam(simple_net.parameters(), lr=0.001)  # 方式一：为不同层定义统一的学习率
optimizer = Adam([{"params":simple_net.hidden.parameters(), "lr":0.1},
                  {"params":simple_net.regression.parameters(), "lr":0.01}], lr=0.001)  # 方式二：为不同层定义不同的学习率

# 参数更新的一般方式
# for batch_x, batch_y in data_loader:
#     optimizer.zero_grad()                   # 梯度清零
#     output = simple_net.forward(batch_x)    # 预测值
#     loss = loss(output, batch_y)            # 损失值
#     loss.backward()                         # 网络反向传播
#     optimizer.step()                        # 更新网络参数

            