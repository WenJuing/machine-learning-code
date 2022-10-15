# from re import A
# import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
# import tensorflow as tf
# from collections import OrderedDict
# from chapter5.two_layer_net_pro import TwoLayerNet
from common import *


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(x_train[:3])
x_train = x_train.reshape(60000, 1, 28, 28)
print(x_train.shape)
print(x_train[:3])
# x_train = x_train.reshape(-1, 784).astype('float32') / 255  # 样本一维化和归一化
# t_train = np.array(tf.one_hot(y_train, 10))
