# 绘制ROC曲线
from turtle import width
from matplotlib import pyplot as plt

FPR = [0, 0, 0, 0, 1 / 6, 1 / 6, 1 / 6, 2 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1]
TPR = [0, 1 / 6, 2 / 6, 3 / 6, 3 / 6, 4 / 6, 5 / 6, 5 / 6, 1, 1, 1, 1, 1]
ticks = [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1]
plt.xticks(ticks)
plt.yticks(ticks)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.xlabel("假正例率")
plt.ylabel("真正例率")
plt.subplot().set_xlim(0, 1)  # 限制坐标刻度
plt.subplot().set_ylim(0, 1)
plt.grid(alpha=0.5)
plt.plot(FPR, TPR)
plt.show()
