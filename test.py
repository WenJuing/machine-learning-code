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


def add(num1: int, num2: int) -> int:
    return num1 + num2

print(add.__annotations__)