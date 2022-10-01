# 用于分类问题的softmax激活函数
import numpy as np


def softmax(a):
    """softmax函数，存在溢出问题"""
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y


def softmax_pro(a):
    """改良的softmax函数"""
    a = a - np.max(a)
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y


if __name__ == '__main__':
    # 溢出问题
    a = np.array([1010, 1000, 900])
    y1 = softmax(a)
    y2 = softmax_pro(a)
    print(y1)
    print(y2)
    # softmax函数的特性
    b = np.array([0.3, 2.9, 4])
    y = softmax_pro(b)
    print(y)
    print(np.sum(y))
    