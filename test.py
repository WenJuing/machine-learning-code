import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
from torch.nn.functional import softmax 
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import torchvision.models as models
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


x = torch.tensor([1, 2, -3, 4, -5])
relu = nn.ReLU()
print(relu(x))
print(x)    # x 未改变
relu = nn.ReLU(inplace=True)
print(relu(x))
print(x)    # x 改变
