# 激活函数
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
relu = nn.ReLU()
softplus = nn.Softplus()

plt.rcParams['axes.unicode_minus']=False
x = torch.linspace(-6, 6, 100)
plt.subplot(221)
plt.plot(x, sigmoid(x))
plt.title("sigmoid")
plt.subplot(222)
plt.plot(x, tanh(x))
plt.title("tanh")
plt.subplot(223)
plt.plot(x, relu(x))
plt.title("relu")
plt.subplot(224)
plt.plot(x, softplus(x))
plt.title("softplus")
plt.show()