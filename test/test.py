# from re import A
import matplotlib.pyplot as plt
import numpy as np
# from keras.datasets import mnist
# import tensorflow as tf
# from collections import OrderedDict
# from chapter5.two_layer_net_pro import TwoLayerNet


# (x_train, y_train), (x_test, y_test) = mnist.load_data()
a = np.array([13,15,16,16,19,20,20,21,22,22,25,25,25,25,30,33,33,35,35,35,35,36,40,45,46,52,70])
plt.boxplot(a)
plt.title("The box plot of data = {13,15,16,16,19,20,20,21,22,22,25,25,25,25,30,33,33,35,35,35,35,36,40,45,46,52,70}", fontsize=8)
plt.show()