# 公共函数库
import numpy as np


def sigmoid(a):
    """隐藏层激活函数"""
    return 1 / (1 + np.exp(-a))


def softmax(a):
    """输出层激活函数"""
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def mean_squared_error(y, t):
    """均方误差"""
    return np.sum((y - t)**2) / 2


def cross_entropy_error(y, t):
    """交叉熵误差"""
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-5)) / batch_size    


def numerical_diff(f, x):
    """差分近似求导"""
    h = 1e-4
    return (f(x+h)-f(x-h)) / (2*h)


def numerical_gradient(f, x):
    """梯度"""
    if x.ndim == 1:
        x = x.reshape(1, x.size)
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp = x[i, j]
            x[i, j] = tmp + h
            fxh1 = f(x)
            x[i, j] = tmp - h
            fxh2 = f(x)
            grad[i, j] = (fxh1 - fxh2) / (2*h)
            x[i, j] = tmp
        
    return grad
        