# 一元线性回归
from matplotlib import pyplot as plt
import numpy as np


def get_w(x, y):
    """计算w"""
    x_avg = np.mean(x)
    m = len(x)
    w1 = 0  # 分子
    w2_left = 0  # 分母左部分
    w2_right = 0  # 分母右部分
    for i in range(m):
        w1 = w1 + y[i] * (x[i] - x_avg)
    for i in range(m):
        w2_left = w2_left + x[i] * x[i]
    w2_right = (1 / m) * (m * x_avg) ** 2
    w = round(w1 / (w2_left - w2_right), 4)
    return w


def get_b(x, y, w):
    """计算b"""
    m = len(x)
    b = 0
    for i in range(m):
        b = b + y[i] - w * x[i]
    b = round(b / m, 4)
    return b


if __name__ == "__main__":
    N = 100  # 样本个数
    x = np.arange(N)
    y = [0.5 * i + 10 + np.random.randn() * 10 for i in x]
    w = get_w(x, y)
    b = get_b(x, y, w)

    plt.scatter(x, y, s=10)  # 绘制样本点
    x1 = 0
    x2 = N
    plt.plot([x1, x2], [w * x1 + b, w * x2 + b], color="red")  # 绘制线性回归函数
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
