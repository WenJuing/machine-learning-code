# k均值算法
import numpy as np
import matplotlib.pyplot as plt


def k_means_by_L2(data):
    """基于欧式距离的K-maens算法"""
    u1 = data[0]
    u2 = data[1]
    while True:
        c1 = []
        c2 = []
        for i in range(data.shape[0]):
            d1 = np.linalg.norm(data[i]-u1, ord=2)
            d2 = np.linalg.norm(data[i]-u2, ord=2)
            if d1 < d2:
                c1.append(data[i])
            else:
                c2.append(data[i])
        c1 = np.array(c1)
        c2 = np.array(c2)
        new_u1 = np.mean(c1, axis=0)
        new_u2 = np.mean(c2, axis=0)
        
        if (u1!=new_u1).any() and (u2!=new_u2).any():
            u1 = new_u1
            u2 = new_u2
        else:
            break

    return c1, c2

def k_means_by_L1(data):
    """基于哈夫曼距离的K-maens算法"""
    u1 = data[0]
    u2 = data[1]
    while True:
        c1 = []
        c2 = []
        for i in range(data.shape[0]):
            d1 = np.linalg.norm(data[i]-u1, ord=1)
            d2 = np.linalg.norm(data[i]-u2, ord=1)
            if d1 < d2:
                c1.append(data[i])
            else:
                c2.append(data[i])
        c1 = np.array(c1)
        c2 = np.array(c2)
        new_u1 = np.mean(c1, axis=0)
        new_u2 = np.mean(c2, axis=0)
        
        if (u1!=new_u1).any() and (u2!=new_u2).any():
            u1 = new_u1
            u2 = new_u2
        else:
            break

    return c1, c2

data = np.array([[3,4],[3,6],[7,3],[4,7],[3,8],[8,5]])
c1, c2 = k_means_by_L2(data)
print(c1,'\n',c2)

plt.scatter(c1[:,0], c1[:,1])
plt.scatter(c2[:,0], c2[:,1])
plt.show()