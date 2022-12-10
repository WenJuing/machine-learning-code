# 使用一元线性回归预测2021年第一季的的利润和销售额
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


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
    df = pd.read_excel("非洲通讯产品销售数据.xlsx", sheet_name="SalesData")
    df = df[df['季度']==1]
    print(df)
    # N = 100  # 样本个数
    w = get_w(df['销售额'], df['利润'])
    b = get_b(df['销售额'], df['利润'], w)

    # plt.scatter(x, y, s=10)  # 绘制样本点
    # x1 = 0
    # x2 = N
    # plt.plot([x1, x2], [w * x1 + b, w * x2 + b], color="red")  # 绘制线性回归函数
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()
