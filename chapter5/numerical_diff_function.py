# 数值微分求近似导、偏导、梯度、梯度下降法
import numpy as np


def numerical_diff(f, x):
    """数值微分求近似导"""
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def numerical_gradient(f, x):
    """求偏导（梯度）"""
    grad = np.zeros_like(x)
    h = 1e-4
    for i in range(x.size):
        tmp = x[i]
        # 计算f(x+h)
        x[i] = tmp + h
        fxh1 = f(x)
        # 计算f(x-h)
        x[i] = tmp - h
        fxh2 = f(x)
        # 计算偏导
        grad[i] = (fxh1 - fxh2) / (2*h)
        x[i] = tmp
    
    return grad
        

def gradient_descent(f, x, a=0.01, step_num=100):
    """梯度下降法"""
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= a * grad
    return x


def function1(x):
    """函数y = 0.01x*x + 0.1*x"""
    return 0.01*x*x + 0.1*x


def function2(x):
    """函数y = x0^2 + x1^2, 参数x=[x0, x1]"""
    return np.sum(x**2)


if __name__ == '__main__':
    # print(numerical_diff(function1, 5)) # 输出0.1999999999  真实导数：0.2
    # print(numerical_diff(function1, 10)) # 输出0.299999999  真实导数：0.3
    print(numerical_gradient(function2, np.array([3.0, 4.0])))  # 参数必须为浮点数
    print(numerical_gradient(function2, np.array([3, 4])))  # 参数必须为浮点数
    # print(gradient_descent(function2, np.array([-3.0, 4.0]), a=0.1))