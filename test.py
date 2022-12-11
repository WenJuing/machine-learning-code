import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error


a = ['a','b','a','c','b','b']
res = pd.value_counts(a)
print(res)
# b    3
# a    2
# c    1
print(res.index)    # Index(['b', 'a', 'c'], dtype='object')
print(res.values)   # [3 2 1]