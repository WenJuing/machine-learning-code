import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil
import time

start = time.time()
for i in range(100):
    print("*")
end = time.time()
print(end-start)