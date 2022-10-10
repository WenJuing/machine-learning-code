# 测试batchnorm方法的影响
from common import *
from multi_layer_net import MultiLayerNet
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255  # 样本一维化和归一化
t_train = np.array(tf.one_hot(t_train, 10))     # 转为标记向量

loss_list = []
bn_loss_list = []
train_acc_list = []
bn_train_acc_list = []

# 超参数
iters_num = 2000         # 更新次数
learning_rate = 0.01       # 学习率
batch_size = 100         # mini_batch
train_size = x_train.shape[0]

optimizer = SGD()

network = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100], output_size=10)
bn_network = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100], output_size=10, use_batchnorm=True)

for i in range(iters_num):
    # 每次随机选取mini_batch进行梯度计算，称为随机梯度下降法（SGD）
    batch_mask = np.random.choice(train_size, batch_size)   
    x_train_batch = x_train[batch_mask]
    t_train_batch = t_train[batch_mask]
    
    
    # 更新参数
    for _network in (network, bn_network):
        grad = _network.gradient(x_train_batch, t_train_batch)
        _network.params = optimizer.update(_network.params, grad)

    loss = network.loss(x_train_batch, t_train_batch)
    loss_list.append(loss)
    bn_loss = bn_network.loss(x_train_batch, t_train_batch)
    bn_loss_list.append(bn_loss)
    
    # 每训练一轮算一次正确率
    if i % batch_size == 0:
        train_acc = network.accuracy(x_train, t_train)
        train_acc_list.append(train_acc)
        bn_train_acc = bn_network.accuracy(x_train, t_train)
        bn_train_acc_list.append(bn_train_acc)
        print("第", (i/batch_size)+1, "轮：", "train_acc:", train_acc, "-"*50)
    print("第",i+1,"次更新：", "loss=", loss)

plt.subplot(121)
plt.plot(np.arange(iters_num), loss_list, label="normal")
plt.plot(np.arange(iters_num), bn_loss_list, label="batchnorm")
plt.xlabel("训练次数")
plt.ylabel("损失值")
plt.subplot(122)
plt.plot(np.arange(len(train_acc_list)), train_acc_list, label="normal")
plt.plot(np.arange(len(bn_train_acc_list)), bn_train_acc_list, label="batchnorm")
plt.legend()
plt.show()