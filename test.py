import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil


shutil.rmtree("./testdir")
os.mkdir("./testdir")