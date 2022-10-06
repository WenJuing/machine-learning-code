# 梯度确认
from two_layer_net_pro import TwoLayerNet
import numpy as np
from keras.datasets import mnist
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255
y_train = np.array(tf.one_hot(y_train, 10))
network = TwoLayerNet(784, 50, 10)
x_batch = x_train[:3]
t_batch = y_train[:3]

grad1 = network.numerical_gradient(x_batch, t_batch)
grad2 = network.gradient(x_batch, t_batch)
for key in grad1.keys():
    diff = np.average(np.abs(grad1[key] - grad2[key]))
    print(key, ":", diff)
print(np.sum(grad1['W1']))
print(np.sum(grad2['W1']))
