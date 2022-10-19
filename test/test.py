# from re import A
# import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
# import tensorflow as tf
# from collections import OrderedDict
# from chapter5.two_layer_net_pro import TwoLayerNet
from common import *


# (x_train, y_train), (x_test, y_test) = mnist.load_data()
A = np.array([[1/3, -1/(3*np.sqrt(2)), -1j/np.sqrt(6)],
              [-1/(3*np.sqrt(2)), 1/6, 1j/(2*np.sqrt(3))],
              [1j/np.sqrt(6), -1j/(2*np.sqrt(3)), 1/2]])

_A = A.T.conjugate()
print(A,"\n")
print(_A,"\n")
print(np.dot(A, _A),"\n")
print(np.dot(_A, A))