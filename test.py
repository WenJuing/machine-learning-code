import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


a = [1, 2, 3, 4, 9, 3]  # 列表
print(a.index(max(a)))
a = np.array([1, 2, 3, 4, 9, 3])    # 数组
print(np.argmax(a))
