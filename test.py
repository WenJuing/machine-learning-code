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


if 'head' in "head.bias":
    print(1)
else:
    print(0)
    