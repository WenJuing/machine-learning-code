# 过拟合现象
from multi_layer_net import MultiLayerNet
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from common import *


(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255  # 样本一维化和归一化
t_train = np.array(tf.one_hot(t_train, 10))     # 转为标记向量
x_test = x_test.reshape(-1, 784).astype('float32') / 255
t_test = np.array(tf.one_hot(t_test, 10))

train_acc_list = []     # 训练集识别正确率
test_acc_list = []      # 测试集识别正确率
# 超参数
iters_num = 1000         # 更新次数
learning_rate = 0.01       # 学习率
batch_size = 100         # mini_batch
train_size = x_train.shape[0]

optimizer = SGD(lr=learning_rate)

# 只用300个数据进行训练和使用7层神经网络，创造过拟合条件
x_train_batch = x_train[:300]
t_train_batch = t_train[:300]
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10)
for i in range(iters_num):

    grad = network.gradient(x_train_batch, t_train_batch)
    
    # 更新参数
    network.params = optimizer.update(network.params, grad)

    # 每训练一轮算一次正确率
    if i % batch_size == 0:
        train_acc = network.accuracy(x_train_batch, t_train_batch)
        train_acc_list.append(train_acc)
        test_acc = network.accuracy(x_test, t_test)
        test_acc_list.append(test_acc)
        print("第", (i/batch_size)+1, "轮：", "train_acc:", train_acc, "test_acc:", test_acc, "-"*50)
    print("第",i+1,"次更新")

plt.plot(np.arange(len(train_acc_list)), train_acc_list, label="训练集")
plt.plot(np.arange(len(test_acc_list)), test_acc_list, label="测试集")
plt.text(len(train_acc_list)-1, train_acc_list[-1], train_acc_list[-1])
plt.text(len(test_acc_list)-1, test_acc_list[-1], test_acc_list[-1])
plt.xlabel("训练代次数")
plt.ylabel("正确率")
plt.legend()
plt.show()