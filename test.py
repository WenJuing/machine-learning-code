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


def main(args):
    print(args.name)	# leimu
    print(getattr(args, 'age', 999))		# 16
    print(args.height)	# 160.5


if __name__ == '__main__':
    # 创建解析器
    parse = argparse.ArgumentParser(description='people')
    # 添加参数
    parse.add_argument('--name', type=str, default='leimu')
    # parse.add_argument('--age', type=int, default=16)
    parse.add_argument('--height', type=float, default=160.5)
    # 解析参数
    args = parse.parse_args()

    main(args)