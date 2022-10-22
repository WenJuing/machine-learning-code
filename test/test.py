# from re import A
# import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
# import tensorflow as tf
# from collections import OrderedDict
# from chapter5.two_layer_net_pro import TwoLayerNet
from common import *


# (x_train, y_train), (x_test, y_test) = mnist.load_data()
a = 2
print(isinstance(a, int))   # True
print(isinstance(a, float)) # False
print(isinstance(a, (int, float))) # True