# 使用单个学习机实现或门、与门、与非门
import numpy as np


def or_gate(x1, x2):
    """或门"""
    x = np.array([x1, x2, 1])
    w = np.array([0.5, 0.5, -0.2])  # 阈值0.2，容易激活
    res = np.dot(w, x.T)
    if res > 0:
        return 1
    else:
        return 0


def and_gate(x1, x2):
    """与门"""
    x = np.array([x1, x2, 1])
    w = np.array([0.5, 0.5, -0.7])  # 阈值0.7，较难激活
    res = np.dot(w, x.T)
    if res > 0:
        return 1
    else:
        return 0


def nand_gate(x1, x2):
    """与非门"""
    x = np.array([x1, x2, 1])
    w = np.array([-0.5, -0.5, 0.7])  # 参数和与门相反
    res = np.dot(w, x.T)
    if res > 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    x1 = 1
    x2 = 0
    print("或门：输入", x1, x2, "输出", or_gate(x1, x2))
    print("与门：输入", x1, x2, "输出", and_gate(x1, x2))
    print("与非门：输入", x1, x2, "输出", nand_gate(x1, x2))