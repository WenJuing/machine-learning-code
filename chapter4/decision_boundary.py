import csv
from email import header
from turtle import color
from sklearn import preprocessing
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(clf,X, y, axes=[0,7.5,0,3], plot_training=True):
    '''绘制决策边界'''
    xls = np.linspace (axes[0],axes[1],100)
    x2s = np.linspace(axes[2], axes[3],100)
    x1,x2 = np.meshgrid(xls,x2s)
    X_new = np.c_[x1.ravel(),x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    plt.contour(x1,x2,y_pred,alpha=0.3,colors='red')
    
    if plot_training:
        plt.plot(X[:, 0][y==0],X[:,1][y==0], "bo")
        plt.plot(X[:, 0][y==1],X[:,1][y==1], "g+")
        plt.axis(axes)
        
        
data = open('watermelon3.0.csv', 'r', encoding='utf8')
reader = csv.reader(data)   # 读取数据
header = next(reader)   # 读取表头
feature_list = np.zeros((17, 2))
result_list = []
i = 0
for row in reader:  # 遍历每行
    result_list.append(row[-1])
    feature_list[i] = np.array(row[1:-1])
    i = i + 1  

X = feature_list
y = preprocessing.LabelBinarizer().fit_transform(result_list)
y = y.ravel()

tree_clf1 = tree.DecisionTreeClassifier(random_state=42)
tree_clf2 = tree.DecisionTreeClassifier(min_samples_leaf=2, random_state=42)
tree_clf1.fit(X, y)
tree_clf2.fit(X, y)
plt.figure(figsize=(12,4))
plt.subplot (121)
plot_decision_boundary(tree_clf1,X, y, axes=[0,0.8,0,0.6])
plt.title('no restriction')
plt.subplot(122)
plot_decision_boundary(tree_clf2,X,y, axes=[0,0.8,0,0.6])
plt.title('min_samples_leaf=2')
plt.show()