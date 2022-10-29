# from re import A
# import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
# import tensorflow as tf
# from collections import OrderedDict
# from chapter5.two_layer_net_pro import TwoLayerNet


# (x_train, y_train), (x_test, y_test) = mnist.load_data()
a = np.arange(12).reshape(3, 4)
print(a.shape)
a = a.reshape(-1, 1)
print(a.shape)