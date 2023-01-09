import torch
from conv_net import ConvNet
from torch.optim import Adam
import torch.nn as nn
from commom import get_MNIST_loader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


conv_net = ConvNet()
optimizer = Adam(conv_net.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()
data_loader, test_x, test_y = get_MNIST_loader()

print_step = 100
loss_list = []
train_acc_list = []
test_acc_list = []

if __name__ == '__main__':
    i = 0
    epoch = 10
    all_step = epoch * 60000.0
    for epoch in range(epoch):
        for step, (train_x, train_y) in enumerate(data_loader):
            y = conv_net(train_x)
            loss = loss_func(y, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            i += 128
            
            if step % print_step == 0:
                _, y = torch.max(y, 1)                  # 返回最大值和下标
                train_acc = accuracy_score(train_y, y)  # accuracy_score(真实标记，预测标记)
                train_acc_list.append(train_acc)
                
                y = conv_net(test_x)
                _, y = torch.max(y, 1)
                test_acc = accuracy_score(test_y, y)
                test_acc_list.append(test_acc)
                print("progress rate", np.around(i/all_step, 5)*100, "%: loss=", np.around(loss.item(), 3), "train acc=", train_acc, "test acc=", test_acc)
            
    plt.subplot(121)
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.subplot(122)
    plt.plot(np.arange(len(train_acc_list)), train_acc_list, label="train")
    plt.plot(np.arange(len(test_acc_list)), test_acc_list, label="test")
    plt.xlabel("per 100 iters")
    plt.ylabel("accuracy")
    plt.show()