# 比较SGD、Momentum和AdaGrad的更新效率
from two_layer_net_pro import TwoLayerNet
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from common import *


(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255  # 样本一维化和归一化
x_test = x_test.reshape(-1, 784).astype('float32') / 255
t_train = np.array(tf.one_hot(t_train, 10))     # 转为标记向量
t_test = np.array(tf.one_hot(t_test, 10))


loss_list = {}
loss_list['SGD'] = []
loss_list['Momentum'] = []
loss_list['AdaGrad'] = []
optimizer = {}
optimizer['SGD'] = SGD()
optimizer['Momentum'] = Momentum()
optimizer['AdaGrad'] = AdaGrad()

# 超参数
iters_num = 2000         # 更新次数
learning_rate = 0.01       # 学习率
batch_size = 100         # mini_batch
train_size = x_train.shape[0]

network = TwoLayerNet(input_size=784, hidden_size=10, output_size=10)
params = {}
params['SGD'] = network.params
params['Momentum'] = network.params
params['AdaGrad'] = network.params

for i in range(iters_num):
    # 每次随机选取mini_batch进行梯度计算，称为随机梯度下降法（SGD）
    batch_mask = np.random.choice(train_size, batch_size)   
    x_train_batch = x_train[batch_mask]
    t_train_batch = t_train[batch_mask]
    
    grad = network.gradient(x_train_batch, t_train_batch)
    # 更新参数
    for key in params.keys():
        params[key] = optimizer[key].update(params[key], grad)
        network.params = params[key]
        loss = network.loss(x_train_batch, t_train_batch)
        loss_list[key].append(loss)
        
    print("第",i+1,"次更新：", "loss=", loss)
    
for key in params.keys():
    plt.plot(np.arange(iters_num), loss_list[key], label=key)
plt.legend()
plt.xlabel("训练次数")
plt.ylabel("损失值")
plt.legend()
plt.show()