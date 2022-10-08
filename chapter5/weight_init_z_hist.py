# 探究使用不同标准型正态分布生成的初始权重矩阵对激活值分布造成的影响
from platform import node
from common import *
from layer import *
import numpy as np
import matplotlib.pyplot as plt


x = np.random.randn(1000, 100)   # 数据集
node_num = 100                  # 每层结点数
hidden_layer_size = 5           # 隐藏层层数
activations = {}                # 保存各层的激活值
relu = Relu()
# 计算各层激活值并保存
for i in range(hidden_layer_size):
    # 若不是第一层，则令x为上一层的激活值
    if i != 0:
        x = activations[i-1]
    # w = np.random.randn(node_num, node_num) * 1.0   # 由标准差为1的正态分布生成的w
    # w = np.random.randn(node_num, node_num) * 0.01   # 由标准差为1的正态分布生成的w
    # w = np.random.randn(node_num, node_num) / np.sqrt(node_num)   # 选Xavier初始值确定的w
    w = np.random.randn(node_num, node_num) * np.sqrt(2/node_num)   # 激活函数为relu时，选He初始值
    z = np.dot(x, w)
    a = relu.forward(z)     # 激活函数为relu
    # a = sigmoid(z)
    # a = np.tanh(z)      # 激活函数用tanh，a的直方图呈现漂亮的吊钟型结构
    activations[i] = a

plt.subplots_adjust(wspace=0.5)
for i, a in activations.items():
    plt.subplot(1, hidden_layer_size, i + 1)
    plt.hist(a.ravel(), 30, range=(0, 1))
    plt.ylim([0, 4e3])
    plt.title("layer-" + str(i+1))
plt.show()
    