# 训练简单 神经网络并保存网络和参数
import torch
from simple_net import SimpleNet
from torch.optim import Adam
from commom import get_diabetes_loader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

data_loader = get_diabetes_loader()
net = SimpleNet()
optimizer = Adam(net.parameters(), lr=0.01)
loss_func = nn.MSELoss()
loss_list = []

if __name__ == '__main__':
    # 训练模型
    for epoch in range(50):
        for batch_x, batch_y in data_loader:
            y = net(batch_x).flatten()
            loss = loss_func(y, batch_y)
            optimizer.zero_grad()           # 梯度初始化为0
            loss.backward()
            optimizer.step()                # 更新参数
            loss_list.append(loss.item())
        print("the", epoch+1, "epoch: loss=", loss.item())

    torch.save(net, "./data/model_and_params/simple_net.pkl")       # 保存整个模型
    network = torch.load("./data/model_and_params/simple_net.pkl")  # 读取整个模型
    print(network)
    torch.save(network.state_dict(), "./data/model_and_params/simple_net_params.pkl")   # 保存模型参数
    params = net = torch.load("./data/model_and_params/simple_net_params.pkl")          # 读取模型参数
    print(params)

    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.show()