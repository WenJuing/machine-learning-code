# 多元线性回归（三元为例）
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def get_W(X, y):
    """计算W"""
    W = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return W


def get_ax():
    """获得3D坐标"""
    # 建立3维坐标
    fig = plt.figure()
    ax = Axes3D(fig)

    return ax


if __name__ == "__main__":
    N = 100  # 样本个数
    X = np.zeros((N, 3))
    X[:, 2] = 1
    X[:, 0] = np.arange(N)  # 设置x
    np.random.seed(10)
    X[:, 1] = [round(0.5 * i + 10 + np.random.randn() * 10, 4) for i in X[:, 0]]  # 设置y
    # 设置z
    Z = np.array([round(0.5 * i + 10 + np.random.randn() * 10, 4) for i in X[:, 0]])

    ax = get_ax()
    W = get_W(X, Z.T)  # W = [w1, w2, b]，其中线性回归 y = w1x1 + w2x2 + b = x^TW
    ax.scatter3D(X[:, 0], X[:, 1], Z)  # 绘制3D散点图

    x1 = 0
    y1 = 0
    z1 = W[0] * x1 + W[1] * y1 + W[2]
    x2 = 100
    y2 = 100
    z2 = W[0] * x2 + W[1] * y2 + W[2]
    ax.plot3D([x1, x2], [y1, y2], [z1, z2], color="red")
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_zlabel("z")
    plt.show()
