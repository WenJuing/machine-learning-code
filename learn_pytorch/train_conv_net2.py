import torch
import pandas as pd
from conv_net2 import ConvNet
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from commom import get_FashionMNIST_loader, train_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    convnet = ConvNet()
    data_loader, X_test, y_test = get_FashionMNIST_loader()
    optimizer = Adam(convnet.parameters(), lr=0.001)
    loss_func = CrossEntropyLoss()
    train_rate = 0.8
    epochs = 3
    # convnet = train_model(convnet, data_loader, train_rate, loss_func, optimizer, epochs)   # 训练模型
    # torch.save(convnet, "./data/model_and_params/convnet2.pkl")
    convnet = torch.load("./data/model_and_params/convnet2.pkl")
    
    # 在测试集上测试模型的泛化能力
    output = convnet(X_test)
    pre = torch.argmax(output, 1)
    test_acc = accuracy_score(pre, y_test)
    print("test acc=", test_acc)
    # 计算混淆矩阵并可视化
    conf_mat = confusion_matrix(pre, y_test)
    class_label = [0,1,2,3,4,5,6,7,8,9]
    df = pd.DataFrame(conf_mat, index=class_label, columns=class_label)
    heatmap = sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('手写数字识别混淆矩阵')
    plt.show()
    