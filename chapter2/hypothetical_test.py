# 假设检验：验证泛化错误率和测试错误率的关系
from matplotlib import pyplot as plt
from scipy.special import comb


def p_of_each_error_num(p_all, m):
    """计算错k个的概率"""
    p_test = []
    for k in range(m + 1):
        p = comb(m, k) * p_all**k * (1 - p_all) ** (m - k)  # 错k个的概率
        p = round(p, 4)
        p_test.append(p)
    return p_test


if __name__ == "__main__":
    p_all = 0.3  # 泛化错误率
    m = 10  # 样本数量
    p_test = p_of_each_error_num(p_all, m)

    plt.bar(range(11), p_test)
    plt.xticks(range(11))
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.xlabel("误分类样本数")
    plt.ylabel("概率")
    plt.show()
