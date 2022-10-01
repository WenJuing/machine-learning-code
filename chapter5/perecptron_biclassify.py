# 使用感知机对两类模式进行分类
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_training_set():
    '''获得鸢尾花训练集'''
    iris_data = pd.read_csv("./data/iris.data")
    iris_data = np.array(iris_data)[:99]
    iris_data[iris_data == 'Iris-setosa'] = 0
    iris_data[iris_data == 'Iris-versicolor'] = 1
    iris_data = np.delete(iris_data, [2, 3], axis=1)
    iris_data = np.insert(iris_data, 2, 1, axis=1)

    return iris_data


def learn_weight(training_set, w):
    '''通过感知机学习权重'''
    data_num = training_set.shape[0]
    a = 0.1
    i = 0
    for n in range(data_num * 500):  # 训练样本个数的500倍次
        out_put = np.dot(training_set[i, :3], w)[0]
        out_put = 1 / (1 + np.exp(-out_put))
        w = w + a * (training_set[i, 3] - out_put) * training_set[i, :3].reshape(3, 1)
        i = i + 1
        if i == data_num - 1:
            i = 0

    return w


if __name__ == '__main__':
    training_set = get_training_set()
    w = np.ones((3, 1))
    w = learn_weight(training_set, w).ravel()
    plt.scatter(training_set[:49, 0], training_set[:49, 1], c='r', label="模式1")
    plt.scatter(training_set[49:, 0], training_set[49:, 1], c='b', label="模式2")
    plt.legend()
    x1 = [min(training_set[:, 0]), max(training_set[:, 0])]
    plt.plot([x1[0], x1[1]], [-(w[0] * x1[0] + w[2]) / w[1], -(w[0] * x1[1] + w[2]) / w[1]])
    plt.title("训练次数为样本量的500倍")
    plt.show()