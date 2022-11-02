import numpy as np

# age
a = np.array([13,15,16,18,19,20,20,21,22,22,25,25,25,30,33,33,35,35,36,40,45,46,52,70])
print(len(a) / 3)   # 多少箱
# 使用最小最大规范化归一化
normal_a = (a-np.min(a)) / (np.max(a)-np.min(a))
for i in normal_a:
    print(str(np.around(i,2))+",",end='')
# 使用z-scores规范化，其中age的标准差为12.94
z_a = (a - np.mean(a)) / 12.94
for i in normal_a:
    print(str(np.around(i,2))+",",end='')
