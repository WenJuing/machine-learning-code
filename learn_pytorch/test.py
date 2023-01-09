import torch.nn as nn
import torch
import numpy as np
import torchvision.models as model
from thop import profile
from ResNet import ResNet101
from commom import compute_FLOPs_and_Params
import os
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image
import matplotlib.pyplot as plt
from ResNet import ResNet101
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# nltk.download('stopwords')
resnet50 = ResNet101()
input = torch.randn(1, 3, 224, 224)
compute_FLOPs_and_Params(resnet50, input)
