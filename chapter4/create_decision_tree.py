import csv
from email import header
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree
import pydotplus


data = open('watermelon2.0.csv', 'r', encoding='utf8')
reader = csv.reader(data)   # 读取数据
header = next(reader)   # 读取表头

feature_list = []
result_list = []
for row in reader:  # 遍历每行
    result_list.append(row[-1])
    feature_list.append(dict(zip(header[1:-1], row[1:-1])))     # 去掉首尾两列的特征集
    
vec = DictVectorizer()
# 给属性编号，均以根据字母升序排序
# color | knock | navel | root | stripe | touch
# 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 
dummyX = vec.fit_transform(feature_list).toarray()
dummyY = preprocessing.LabelBinarizer().fit_transform(result_list)
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
clf = clf.fit(dummyX, dummyY)

dot_data = tree.export_graphviz(clf,
                                feature_names=vec.get_feature_names_out(),
                                filled=True, rounded=True,
                                special_characters=True,
                                out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("watermelon.pdf")

# test = [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0]     # 测试样例，参数以编号表示
# predict_result = clf.predict(test)