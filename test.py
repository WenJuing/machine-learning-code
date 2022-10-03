from re import A
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import tensorflow as tf

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
a = np.arange(12).reshape(3,4)
print(a[[0, 1]])
