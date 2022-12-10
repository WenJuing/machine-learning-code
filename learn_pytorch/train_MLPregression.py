import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from MLPregression import MLPregression
from commom import get_california_loader
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np


mlp = MLPregression()
data_loader, X_test, y_test = get_california_loader()
optimizer = Adam(mlp.parameters(), lr=0.001)
loss_func = nn.MSELoss()    # 回归问题中使用均方误差损失函数
writer = SummaryWriter(log_dir='./data/train_MLPregression_log')

epochs = 30
if __name__ == '__main__':
    # for epoch in range(epochs):
    #     for step, (X_train, y_train) in enumerate(data_loader):
    #         output = mlp(X_train)
    #         loss = loss_func(output, y_train)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     print("epoch:", epoch+1, "loss=", loss.item())
    #     writer.add_scalar("Loss", loss.item(), epoch+1)
        
    # writer.close()
    # torch.save(mlp, "./data/model_and_params/MLPregression.pkl")
    mlp = torch.load("./data/model_and_params/MLPregression.pkl")
    # 计算预测误差
    pre_y = mlp(X_test).detach().numpy()    # size=(6192,)
    error = mean_absolute_error(pre_y, y_test)
    print(error)    # 0.37，预测效果比较理想
    # 可视化真实值和预测值
    index = np.argsort(y_test)  # 返回值从小到大的索引
    plt.scatter(np.arange(len(pre_y)), pre_y[index], s=3, label="预测值")
    plt.scatter(np.arange(len(pre_y)), y_test[index], s=3, label="真实值")
    plt.legend()
    plt.show()