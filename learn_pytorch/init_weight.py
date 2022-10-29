# 权重初始化
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


conv2 = nn.Conv2d(3, 16, 3)
torch.manual_seed(1)
nn.init.normal(conv2.weight, mean=0, std=1)  # 初始化权重（标准正态分布尺度）
nn.init.constant(conv2.bias, val=0.1)        # 初始化偏置（值全为0.1）

print(conv2.weight.data.shape)  # (16,3,3,3)
print(conv2.bias.shape)         # (16,)

plt.rcParams['axes.unicode_minus']=False
plt.hist(conv2.weight.data.numpy().ravel(), bins=30)
plt.show()