import torch
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from torchvision.datasets import FashionMNIST, MNIST, ImageFolder
from sklearn.datasets import load_diabetes, load_boston, fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from thop import profile
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
from PIL import Image


# 分类数据
def get_FashionMNIST_loader():
    # train  (60000,1,28,28)   x:(28,28)
    # test   (10000,1,28,28)   t:0~9
    data = FashionMNIST('./data/FashionMNIST', train=True, transform=transforms.ToTensor(), download=False)
    data_loader = Data.DataLoader(dataset=data, batch_size=128, shuffle=False, num_workers=2)
    
    test_data = FashionMNIST('./data/FashionMNIST', train=False, transform=transforms.ToTensor(), download=False)
    X_test = test_data.data.float()
    X_test = torch.unsqueeze(X_test, dim=1)
    y_test = test_data.targets
    
    return data_loader, X_test, y_test


def get_MNIST_loader():
    # train  (60000,1,28,28)   x:(28,28)
    # test   (10000,1,28,28)   t:0~9
    train_data = MNIST('./data/MNIST', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=1)
    
    test_data = MNIST('./data/MNIST', train=False, transform=transforms.ToTensor(), download=False)
    X_test = test_data.data.float()  # X_test.shape = [10000, 28, 28]
    X_test = torch.unsqueeze(X_test, dim=1)     # 添加通道维数，X_test.shape = [10000, 1, 28, 28]
    y_test = test_data.targets  # y_test.shape = [10000]
    
    return data_loader, X_test, y_test


def get_spambase(test_size=0.25):
    # 1 垃圾邮件 1813
    # 0 非垃圾邮件 2788
    spam = np.array(pd.read_csv("./data/spambase.data", header=None))   # 对于没有表头的数据集，header设置为None
    # 将数据随机切分为训练集和数据集
    x_train, x_test, y_train, y_test = train_test_split(spam[:,:-1], spam[:,-1], test_size=test_size, random_state=123)
    # 使用最大-最小方法对数据进行归一化（数据预处理很重要！！）
    scale = MinMaxScaler(feature_range=(0,1))   # 缩放尺度，默认0~1
    x_train = scale.fit_transform(x_train)      # fit本质求min和max，用过一次后后面transform不用再fit
    x_test = scale.transform(x_test)
    
    x_train = torch.as_tensor(x_train).float()
    y_train = torch.as_tensor(y_train).long()
    train_data = Data.TensorDataset(x_train, y_train)
    train_data_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=1)
    
    x_test = torch.as_tensor(x_test).float()
    y_test = torch.as_tensor(y_test).long()

    return train_data_loader, x_test, y_test


def get_IMDA_loader(test_ratio=0.25):
    """IMDB 50k movie review
        50k review (str)
        50k sentiment (positive/negative)
    """
    # 若没有预处理数据，则读入原始数据并进行预处理后保存
    if (os.path.exists("./data/IMDB_50k/imdb_train.csv") and os.path.exists("./data/IMDB_50k/imdb_test.csv")) == False:
        imda = np.array(pd.read_csv("./data/IMDB Dataset.csv"))
        train_text, test_text, train_label, test_label = train_test_split(imda[:, :-1], imda[:, -1], test_size=test_ratio, random_state=100)
        train_text = train_text.flatten()
        test_text = test_text.flatten()
        # 将 postive转换成1，negative转换成0
        train_label[train_label=='positive'], train_label[train_label=='negative'] = 1, 0
        test_label[test_label=='positive'], test_label[test_label=='negative'] = 1, 0
        # 数据预处理
        stop_words = stopwords.words("english")
        stop_words = set(stop_words)
        train_text_pre = text_preprocess(train_text)
        train_text_pre = del_stopwords(train_text_pre, stop_words)
        test_text_pre = text_preprocess(test_text)
        test_text_pre = del_stopwords(test_text_pre, stop_words)
        # 将处理好的数据保存到csv，以便下次直接使用
        texts = [" ".join(words) for words in train_text_pre]
        train_datasave = pd.DataFrame({"text": texts, "label": train_label})
        texts = [" ".join(words) for words in test_text_pre]
        test_datasave = pd.DataFrame({"text": texts, "label": test_label})
        train_datasave.to_csv("./data/IMDB_50k/imdb_train.csv", index=False)
        test_datasave.to_csv("./data/IMDB_50k/imdb_test.csv", index=False)
    else:
        print("弃坑了")
        
        
# 回归数据
def get_diabetes_loader():
    # X_train   (442,10)  float64
    # y_train   (442,)    float64
    X_train, y_train = load_diabetes(return_X_y=True)
    # 数据标准化处理
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    
    X_train = torch.as_tensor(X_train).float()
    y_train = torch.as_tensor(y_train).float()
    train_data = Data.TensorDataset(X_train, y_train)
    data_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=1)
    
    return data_loader

def get_boston_loader():
    # X_train   (506,13)  float64
    # y_train   (506,)    float64
    X_train, y_train = load_boston(return_X_y=True)
    # 数据标准化处理
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    
    X_train = torch.as_tensor(X_train).float()
    y_train = torch.as_tensor(y_train).float()
    train_data = Data.TensorDataset(X_train, y_train)
    data_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=1)
    
    return data_loader


def get_california_loader():
    housedata = fetch_california_housing()
    # X_train.shape: (14448, 8)
    X_train, X_test, y_train, y_test = train_test_split(housedata.data, housedata.target, test_size=0.3, random_state=100)
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)
    # 查看相关系数热力图
    # df = pd.DataFrame(data=X_train, columns=housedata.feature_names)
    # df['target'] = y_train
    # show_corrcoef(df)
    X_train = torch.as_tensor(X_train).float()
    y_train = torch.as_tensor(y_train).float()
    X_test = torch.as_tensor(X_test).float()
    y_test = torch.as_tensor(y_test).float()
    
    train_data = Data.TensorDataset(X_train, y_train)
    data_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=1)
    
    return data_loader, X_test, y_test

def get_flower_loader(batch_size=10):
    # data size: 3670
    # class size: 5
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = ImageFolder("./data/flower_photos", transform=data_transforms)
    data_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    
    return data_loader


def preprocess_image_to_input(image):
    """将一张图片处理成符合神经网络输入的形式"""
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # 使用transforms时，图片要求是PIL Image类型
    im_input= data_transforms(image).unsqueeze(0)
    
    return im_input


# 实用工具
def show_data(data_loader):
    """显示一个batch size的图片"""
    for batch_x, batch_y in data_loader:
        break
    batch_size = len(batch_x)
    row = int(np.ceil(batch_size/16))
    batch_x = batch_x.squeeze()
    for i in range(batch_size):
        plt.subplot(row, 16, i+1)
        plt.imshow(batch_x[i], cmap=plt.cm.gray)
        plt.title(batch_y[i].item(), size=9)
        plt.axis("off")
        plt.subplots_adjust(hspace=0.05,wspace=0.05)
    plt.show()


def show_corrcoef(df):
    """绘制相关系数(correlation coefficient)热力图"""
    datacor = np.corrcoef(df.values, rowvar=0)  # rowvar默认为True，即每一行为一个变量（观测值），这里每一列为一个变量
    datacor = pd.DataFrame(data=datacor, columns=df.columns, index=df.columns)
    
    plt.figure(figsize=(8, 6))
    plt.rcParams['axes.unicode_minus']=False    # 正常显示负号
    ax = sns.heatmap(datacor, square=True, annot=True, fmt=".3f", linewidths=.5, cmap="YlGnBu", 
                     cbar_kws={"fraction":0.05, "pad": 0.05})
    plt.title("相关系数热力图")
    plt.show()
    

def train_model(model, data_loader, loss_function, optimizer, epochs, device, 
                model_name=None, dataset_name=None, is_rnn=False):
    model.train()
    sw = SummaryWriter(log_dir="./runs/train_" + model_name)
    accu_loss = torch.zeros(1).to(device)    # 累计损失
    accu_num = torch.zeros(1).to(device)     # 累计预测正确的样本数
    optimizer.zero_grad()
    
    sample_num = 0
    cur_loss = 0
    cur_acc = 0
    print("model:", model_name, "| dataset:", dataset_name, "| device:", device)
    for epoch in range(epochs):
        data_loader = tqdm(data_loader)     # 显示进度条
        for _, (images, labels) in enumerate(data_loader):
            sample_num += images.shape[0]
            # 若模型为rnn，则输入数据 [B, C, H, W] -> [B, time_step, input_dim]=[B, H, W]
            if is_rnn:
                images = images.mean(dim=1)
                # images = images.view(-1, images.shape[2], images.shape[3])
            pred = model(images.to(device))
            pred_classes = torch.argmax(pred, 1)
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()
            
            loss = loss_function(pred, labels.to(device))
            loss.backward()
            accu_loss += loss.detach()
            
            cur_loss, cur_acc = accu_loss.item() / sample_num, accu_num.item() / sample_num
            data_loader.set_description("[train epoch %d] loss: %.3f, acc: %.3f" % (epoch, cur_loss, cur_acc))
            optimizer.step()
            optimizer.zero_grad()
            
        sw.add_scalar(model_name + "/" + dataset_name + " train/loss", cur_loss, epoch)
        sw.add_scalar(model_name + "/" + dataset_name + " train/accuracy", cur_acc, epoch)

    return model


def get_vgg16_feature():
    vgg16 = models.vgg16(pretrained=True)   # 获取预训练的VGG16
    vgg_feature = vgg16.features            # 获取特征提取层（卷积层、池化层，不包括全连接层的参数）
    for param in vgg_feature.parameters():  # 冻结参数，训练过程中不对其更新
        param.requires_grad_(False)
    
    return vgg_feature


def compute_FLOPs_and_Params(model, input_size=(1, 3, 224, 224)):
    """input_size (B, C, H, W): 数据的大小, B不影响计算结果"""
    flops, params = profile(model, (input_size,), verbose=False)
    print("FLOPs: %.1f G" % (flops / 1E9))
    print("Params: %.1f M" % (params / 1E6))


def text_preprocess(text_list):
    """预处理文本数据
    Args:
        text_list (array(str)): text_list.shape is (n,)
    """
    text_list_pre = []
    for text in text_list:
        text = text.lower()
        text = re.sub("<br />", "", text)  # 去除<br />
        text = re.sub("\d+", "", text)     # 去除数字
        text = text.translate(str.maketrans("", "", string.punctuation.replace("'", "")))   # 去除标点符号，'不去除
        text_list_pre.append(text.strip())
    
    return np.array(text_list_pre)


def del_stopwords(text_list, stop_words):
    """删除停用词
    Args:
        text_list (array(str)): text_list.shape is (n,)
    """
    text_list_pre = []
    for text in text_list:
        text_words = word_tokenize(text)    # 对文本进行分词，结果是一个列表
        text_words = [word for word in text_words if word not in stop_words]  # 删除停用词
        # 删除带有'的单词，比如 it's
        text_words = [word for word in text_words if len(re.findall("'", word)) == 0]
        text_list_pre.append(text_words)
        
    return np.array(text_list_pre, dtype=object)
    
    
features = []
def hook(model, input, output):
    features.append(output.detach())


if __name__ == '__main__':
    # data_loader = get_boston_loader()
    # data_loader, X_test, y_test = get_MNIST_loader()
    # data_loader, X_test, y_test = get_FashionMNIST_loader()
    data_loader = get_flower_loader()
    # for step, (X_batch, y_batch) in enumerate(data_loader):
    #     img = X_batch[0]
    #     print(X_batch.shape)
    #     print(y_batch.shape)
    #     print(y_batch)
    #     break