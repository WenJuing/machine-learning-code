# 使用正向传播方程识别手写数字
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import pickle


def get_train_data():
    """获得训练集"""
    # d读取时将图像归一化并展开成一维向量
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255  # 归一化并转一维
    return x_train, y_train


def init_network():
    """初始化神经网络参数"""
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def sigmoid(a):
    """sigmoid激活函数"""
    z = 1 / (1 + np.exp(-a))
    return z


def softmax(a):
    """softmax激活函数"""
    a = a - np.max(a)
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    z = np.around(exp_a / sum_exp_a, 3)
    
    return z


def predict(network, x):
    """类别预测"""
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y


if __name__ == '__main__':
    images, labels = get_train_data()
    network = init_network()
    N = 100    # 识别数量
    accuracy_cnt = 0
    for i in range(N):
        y = predict(network, images[i])
        p = np.argmax(y)    # 获得最大值的索引（刚好对应预测值的真实标记）
        print("[No%d]  识别为:%d  真实为:%d" % (i+1, p, labels[i]))
        if p == labels[i]:
            accuracy_cnt += 1
    accuracy_rate = np.around(accuracy_cnt / N, 4)
    print("预测正确率：", accuracy_rate*100, "%")