import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 导入数据
data = pd.read_csv("./GoodsOrder.csv", encoding="gbk")
types = pd.read_csv("./GoodsTypes.csv", encoding="gbk")

# 1、描述性统计分析
data.info()     # 查看数据属性，id, Goods
print(data['id'].describe())    # 查看id属性的相关统计，id:1~9835（一个id对应多个商品）

# 1.1、缺失值分析
null_count = data.isnull().sum()
# print(null_count)
# 1.2 异常值分析
a = 0
b = 0
for i in data['Goods']:
    if i in list(types['Goods']):
        a += 1
    else:
        b += 1
# print(a,b)
# 43285 0
# 1.3、重复数据分析
res = data.duplicated().sum()
print(res)
# 2、绘制条形图分析热销商品
hot_goods = pd.value_counts(data['Goods'])[:10]     # 获取前十热销商品和统计
hot_goods_rate = np.around(np.array(hot_goods.values) / len(data) * 100, 3) # 计算热销商品占比
hot_goods_name = np.array(hot_goods.index)
hot_goods_count = np.array(hot_goods.values)
# print("商品名称|销量|销量占比")
# for i in range(10):
#     print(hot_goods_name[i]+'|'+str(hot_goods_count[i])+'|'+str(hot_goods_rate[i])+"%")

# plt.barh(hot_goods_name, hot_goods_count)
# plt.title("商品的销量Top10")
# plt.xlabel("销量")
# plt.ylabel("商品名称")
# plt.show()

# 3、分析归类后各类别商品的销量及其占比
group = data.groupby(['Goods']).count().reset_index()
sort = group.sort_values('id', ascending=False).reset_index()
data_nums = data.shape[0]   # 总量
del sort['index']
sort_links = pd.merge(sort, types)  # 根据type合并两个DataFrame
# 根据类别求和,求每个商品类别的总量,并排序
sort_link = sort_links.groupby(['Types']).sum().reset_index()
sort_link = sort_link.sort_values('id', ascending=False).reset_index()
del sort_link['index']  # 删除index列
# 求百分比,然后更换列名,最后输出到文件
sort_link['count'] = sort_link.apply(lambda line: line['id'] / data_nums, axis=1)
sort_link.rename(columns={'count': 'percent'}, inplace=True)
# print("各类别商品的销量及其占比:\n", sort_link)
sort_link.to_csv('./percent.csv', index=False, header=True, encoding='gbk')    # 保存结果
# 绘制饼图展示每类商品销量占比
data1 = sort_link['percent']
labels = sort_link['Types']
plt.pie(data1, labels=labels, autopct='%1.2f%%')
plt.title("各类别商品销量占比")
plt.savefig('./percent.png')
plt.show()

# 4、计算非酒精饮料内部商品的销量及其占比
# 先筛选非酒精饮料类别的商品,求百分比,然后输出结果到文件
selected = sort_links.loc[sort_links['Types']=='非酒精饮料']
child_nums = selected['id'].sum()   # 对所有的非酒精饮料类进行求和
selected['child_percent'] = selected.apply(lambda line: line['id'] / child_nums, axis=1)  # 求百分比
selected.rename(columns={'id': 'count'}, inplace=True)
# print("非酒精饮料内部商品的销量及其占比:\n", selected)
sort_link.to_csv('./child_percent.csv', index=False, header=True, encoding='gbk')
# 绘制饼图展示非酒精饮料内部商品的销量占比
data2 = selected['child_percent']
labels = selected['Goods']
explode = (0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.08,0.3,0.1,0.3) # 设置每一块分割出的间隙大小
plt.pie(data2,explode=explode,labels=labels,autopct='%1.2f%%', pctdistance=1.1, labeldistance=1.2)
plt.title("非酒精饮料内部各商品的销量占比")
plt.axis("equal")
plt.savefig('./child_persent.png')
plt.show()

# 5、数据预处理
# 根据id对Goods列合并,并使用“,”将各商品隔开
data['Goods'] = data['Goods'].apply(lambda x: ','+x)
data = data.groupby('id').sum().reset_index()
# 对合并的商品列转换数据格式
data['Goods'] = data['Goods'].apply(lambda x : [x[1:]])
data_list = list(data['Goods'])
# 分割商品名,每个商品为独立元素
data_translation = []
for i in data_list:
    p = i[0].split(',')
    data_translation.append(p)
print("数据转换结果的前5个元素:\n", data_translation[:5])

# 6、构建关联规则模型
from apriori import *
dataSet = data_translation
L, supportData = apriori(dataSet, minSupport=0.02)
rule = gen_rule(L, supportData, minConf=0.35)