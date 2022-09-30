import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os


data = pd.read_csv("./data/iris.data")

data = np.array(data)
data[data == 'Iris-setosa'] = 0
data[data == 'Iris-versicolor'] = 1
data = np.delete(data, [2,3], axis=1)
data = np.insert(data, 2, 1, axis=1)
print(data[:99])