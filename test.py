import numpy as np
from matplotlib import pyplot as plt


x = np.arange(-40,40)
y = []
for i in x:
    t = np.round(1 / (1 + np.exp(-i)), 4)
    y.append(t)
plt.plot(x,y)
plt.show()