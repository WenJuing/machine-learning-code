import numpy as np
import matplotlib.pyplot as plt

a = np.array([13,15,16,16,19,20,20,21,22,22,25,25,25,25,30,33,33,35,35,35,35,36,40,45,46,52,70])
plt.boxplot(a)
plt.title("The box plot of data = {13,15,16,16,19,20,20,21,22,22,25,25,25,25,30,33,33,35,35,35,35,36,40,45,46,52,70}", fontsize=8)
# plt.show()

x = np.array([22,1,42,10])
y = np.array([20,0,36,8])
dxy = np.sqrt(np.sum((x-y)**2))
mxy = np.sum(np.abs(x-y))
print(dxy)
print(mxy)
print(np.linalg.norm(x-y, ord=2))   # 欧式距离
print(np.linalg.norm(x-y, ord=1))   # 曼哈顿距离
print(np.linalg.norm(x-y, ord=3))   # 闵氏距离
print(np.linalg.norm(x-y, ord=np.inf))   # 上确界距离
