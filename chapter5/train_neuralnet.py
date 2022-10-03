# 训练两层神经网络
from cProfile import label
from two_layer_net import TwoLayerNet
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255  # 样本一维化和归一化
x_test = x_test.reshape(-1, 784).astype('float32') / 255
t_train = np.array(tf.one_hot(t_train, 10))     # 转为标记向量
t_test = np.array(tf.one_hot(t_test, 10))

loss_list = []
train_acc_list = []     # 训练集识别正确率
test_acc_list = []      # 测试集识别正确率
# 超参数
iters_num = 2000         # 更新次数
learning_rate = 0.1       # 学习率
batch_size = 100         # mini_batch
train_size = x_train.shape[0]

network = TwoLayerNet(input_size=784, hidden_size=5, output_size=10)
for i in range(iters_num):
    # 每次训练随机选取数据组成mini_batch，出于需要更多的训练次数和计算量问题，暂时不采用
    batch_mask = np.random.choice(train_size, batch_size)   
    x_train_batch = x_train[batch_mask]
    t_train_batch = t_train[batch_mask]
    
    grad = network.numerical_gradient(x_train_batch, t_train_batch)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_train_batch, t_train_batch)
    loss_list.append(loss)
    
    # 每训练一轮算一次正确率
    if i % batch_size == 0:
        train_acc = network.accuracy(x_train, t_train)
        train_acc_list.append(train_acc)
        test_acc = network.accuracy(x_test, t_test)
        test_acc_list.append(train_acc)
        print("第", (i/batch_size)+1, "轮：", "train_acc:", train_acc, "test_acc:", test_acc, "-"*50)
    print("第",i+1,"次更新：", "loss=", loss)

plt.subplot(121)
plt.plot(np.arange(iters_num), loss_list)
plt.xlabel("训练次数")
plt.ylabel("损失值")
plt.subplot(122)
plt.plot(np.arange(len(train_acc_list)), train_acc_list, label="训练集")
plt.plot(np.arange(len(test_acc_list)), test_acc_list, label="测试集")
plt.legend()
plt.show()