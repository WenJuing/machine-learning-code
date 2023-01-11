import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
from torch.nn.functional import softmax 
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import torchvision.models as models
import argparse
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(log_dir="./runs/example")
for i in range(10):
    writer.add_scalars('model/dataset Accuracy', {'train': i**2, 'test': np.sqrt(i)}, i)
    writer.add_scalars('model/dataset Loss', {'train': 1/(i+1), 'test': 1/(i+2)}, i)
    writer.add_scalar('model/dataset Learning rate', np.exp(np.cos(i)), i)