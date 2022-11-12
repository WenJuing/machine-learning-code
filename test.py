import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil
import time

R = np.arange(6).reshape(2,3)
c = np.array([1,2,3])
print(np.matmul(R,c))
print(np.matmul(c,R))