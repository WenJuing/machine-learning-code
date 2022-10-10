# 比较不同初始值时loss的变化
from two_layer_net_pro import TwoLayerNet
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255  # 样本一维化和归一化
t_train = np.array(tf.one_hot(t_train, 10))     # 转为标记向量

std_loss_list = []
Xavier_loss_list = []
He_loss_list = []

# 超参数
iters_num = 5000         # 更新次数
learning_rate = 0.1       # 学习率
batch_size = 100         # mini_batch
train_size = x_train.shape[0]

network_std = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=0.01)
network_Xavier = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=1/np.sqrt(784))
network_Xavier.params['W2'] = np.random.randn(50, 10) / np.sqrt(50)
network_He = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=0.01)
for i in range(iters_num):
    # 每次随机选取mini_batch进行梯度计算，称为随机梯度下降法（SGD）
    batch_mask = np.random.choice(train_size, batch_size)   
    x_train_batch = x_train[batch_mask]
    t_train_batch = t_train[batch_mask]
    
    # grad = network.numerical_gradient(x_train_batch, t_train_batch)
    grad = network.gradient(x_train_batch, t_train_batch)
    
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_train_batch, t_train_batch)
    std_loss_list.append(loss)
    
    print("第",i+1,"次更新：", "loss=", loss)

plt.subplot(121)
plt.plot(np.arange(iters_num), std_loss_list)
plt.xlabel("训练次数")
plt.ylabel("损失值")
plt.subplot(122)
plt.legend()
plt.show()