# 使用对数几率回归进行分类
from matplotlib import pyplot as plt
import numpy as np


def get_p1(B, x):
    """计算p(y=1|x;B)"""
    e_part = np.exp(np.dot(B.T, x))
    p1 = e_part / (1 + e_part)
    return p1


def get_p0(B, x):
    """计算p(y=0|x;B)"""
    p0 = 1 - get_p1(B, x)
    return p0


def get_fisrt_partial_derivative(B, test):
    """计算l对B的一阶偏导"""
    first_partial = np.zeros((3, 1))
    for i in range(test.shape[1]):
        first_partial = first_partial + test[:3, i].reshape(3, 1) * (
            test[3, i] - get_p1(B, test[:3, i].reshape(3, 1))
        )
    first_partial = -first_partial
    return first_partial


def get_second_partial_derivative(B, test):
    """计算l对B和B^T的二阶偏导"""
    second_partial = np.zeros((3, 3))
    for i in range(test.shape[1]):
        second_partial = second_partial + np.dot(
            test[:3, i].reshape(3, 1), test[:3, i].reshape(1, 3)
        ) * get_p1(B, test[:3, i].reshape(3, 1)) * (
            get_p0(B, test[:3, i].reshape(3, 1))
        )
    return second_partial


def get_B_by_newton_method(B, test):
    """使用牛顿迭代法求l最小时B的解"""
    B2 = B
    B1 = B2 - np.dot(
        np.linalg.inv(get_second_partial_derivative(B2, test)),
        get_fisrt_partial_derivative(B2, test),
    )
    for i in range(10):
        B2 = B1
        B1 = B2 - np.dot(
            np.linalg.inv(get_second_partial_derivative(B2, test)),
            get_fisrt_partial_derivative(B2, test),
        )
    return B1


def generate_practice_dataset(N):
    """生成训练集"""
    N = 80  # 训练集样本个数
    test = np.zeros((4, N))  # 训练集test，test第i列为[xi1,xi2,1,yi]，代表一个样本
    test[2] = 1
    np.random.seed(10)
    for i in range(int(N / 2)):  # 类1
        test[0, i] = np.random.randint(6) / 100
        test[1, i] = np.random.randint(7, 15) / 100
        test[3, i] = 1
    np.random.seed(11)
    for i in range(int(N / 2), N):  # 类2
        test[0, i] = np.random.randint(7, 15) / 100
        test[1, i] = np.random.randint(6) / 100
        test[3, i] = 0

    return test


def draw_logistic_function(B, test):
    """绘制对数几率函数"""
    # 计算横轴序列
    xs = []
    for i in range(N):
        x = np.dot(B.reshape(1, 3), test[:3, i].reshape(3, 1))
        x = np.round(x[0, 0], 2)
        xs.append(x)
    xs = sorted(xs)

    # 计算纵轴序列
    y = []
    for x in xs:
        t = np.round(1 / (1 + np.exp(-x)), 4)
        y.append(t)

    # plt.scatter(test[0], test[1], s=10)     # 绘制训练集

    num1 = len([i for i in y if i >= 0.5])  # 正例个数（大于0.5判断为正数）
    num0 = len([i for i in y if i < 0.5])  # 负例个数（小于0.5判断为负数）
    plt.plot(xs, y, color="red")  # 绘制对数几率函数
    plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
    plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
    plt.xlabel("正例个数："+str(num1)+"  反例个数："+str(num0))
    plt.show()


if __name__ == "__main__":
    N = 80  # 训练集个数
    test = generate_practice_dataset(N)
    # 给定初始w、b
    B = np.arange(3).reshape(3, 1)
    B = get_B_by_newton_method(B, test)
    draw_logistic_function(B, test)
