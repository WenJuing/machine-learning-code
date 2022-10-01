# 正向传播方程
import numpy as np


def init_network():
    """初始化神经网络参数"""
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network


def forward(network, x):
    """正向传播算法"""
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 输入层到第一层的计算
    A1 = np.dot(x, W1) + b1
    Z1 = sigmoid(A1)

    # 第一层到第二层的计算
    A2 = np.dot(Z1, W2) + b2
    Z2 = sigmoid(A2)

    # 第二层到输出层的计算
    A3 = np.dot(Z2, W3) + b3
    y = A3      # 最后层的激活函数根据具体情况而定
    
    return y


def sigmoid(A):
    """sigmoid激活函数"""
    Z = 1 / (1 + np.exp(-A))
    return Z


if __name__ == '__main__':
    network = init_network()
    x = np.array([1, 0.5])
    y = forward(network, x)
    print(y)
    