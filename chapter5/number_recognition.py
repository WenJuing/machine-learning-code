# 使用正向传播方程识别手写数字
from keras.datasets import mnist
import numpy as np
import pickle


def get_test_data():
    """获得测试集"""
    # 训练集60000张，测试集10000张，均为28×28大小的手写数字图片
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 读取时将图像归一化并展开成一维向量
    x_test = x_test.reshape(-1, 784).astype('float32') / 255
    return x_test, y_test


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
    images, labels = get_test_data()
    network = init_network()
    accuracy_cnt = 0
    batch_size = 100
    for i in range(0, len(images), batch_size): # 一次处理100张图片
        y_batch = predict(network, images[i:i+batch_size])
        accuracy_cnt += np.sum(np.argmax(y_batch, axis=1) == labels[i:i+batch_size])    # 获得最大值的索引（刚好对应预测值的真实标记）
    accuracy_rate = np.around(accuracy_cnt / len(images), 4)
    print("预测正确率：", accuracy_rate*100, "%")