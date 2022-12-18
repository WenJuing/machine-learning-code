import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score


y = [0,1,2]
pre = [0,1,3]
acc = accuracy_score(y, pre)
print(acc)