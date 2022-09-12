# 绘制代价曲线
from turtle import position, right
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def theta_result(output_value, theta):
    """根据阈值对样本进行预测分类"""
    theta_result = []
    for i in range(len(output_value)):
        if output_value[i] < theta:
            theta_result.append(0)
        else:
            theta_result.append(1)
    return theta_result


def count_m_positive_and_negative(sample):
    """统计样本中正例和反例的个数"""
    result = pd.value_counts(sample)
    m_positive = result[1]
    m_negative = result[0]
    return m_positive, m_negative


def count_TP_FP_FN_TN(sample, theta_result):
    """统计真正例、假正例、假反例和真反例"""
    TP = FP = FN = TN = 0
    for i in range(len(sample)):
        if theta_result[i] == 1:  # 预测为正例
            if sample[i] == 1:
                TP += 1
            else:
                FP += 1
        else:  # 预测为反例
            if sample[i] == 1:
                FN += 1
            else:
                TN += 1
    return TP, FP, FN, TN


def get_FNR_FPR(TP, FP, FN, TN):
    """计算真正例率和假正例率"""
    TPR = round(TP / (TP + FN), 4)
    FNR = round(1 - TPR, 4)
    FPR = round(FP / (TN + FP), 4)
    return FNR, FPR


def get_Pcost(p, cost01, cost10):
    """计算正概率代价"""
    Pcosts = []
    for i in range(len(p)):
        Pcost = round((p[i] * cost01) / (p[i] * cost01 + (1 - p[i]) * cost10), 4)
        Pcosts.append(Pcost)
    return Pcosts


def get_cost_norm(p, cost01, cost10, FNR, FPR):
    """计算归一化代价"""
    costs_norm = []
    for i in range(len(p)):
        cost_norm = round(
            (FNR * p[i] * cost01 + FPR * (1 - p[i]) * cost10)
            / (p[i] * cost01 + (1 - p[i]) * cost10),
            4,
        )
        costs_norm.append(cost_norm)
    return costs_norm


def draw_cost_curve(output_value, sample, cost01, cost10, theta):
    """绘制代价曲线"""
    for i in range(len(theta)):
        theta[i] = round(theta[i], 2)
        result = theta_result(output_value, theta[i])  # 根据阈值预测分类
        TP, FP, FN, TN = count_TP_FP_FN_TN(sample, result)
        FNR, FPR = get_FNR_FPR(TP, FP, FN, TN)
        Pcost = get_Pcost(p, cost01, cost10)  # 正例概率代价
        cost_norm = get_cost_norm(p, cost01, cost10, FNR, FPR)  # 归一化代价
        plt.plot(Pcost, cost_norm)

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.xlabel("正例概率代价")
    plt.ylabel("归一化代价")
    plt.legend(theta, loc="upper right")
    plt.subplot().set_xlim(0, 1)
    plt.subplot().set_ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    output_value = [  # 学习器预测值
        0.1,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.6,
        0.65,
        0.7,
        0.8,
        0.9,
        0.95,
    ]
    sample = [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1]  # 样本
    p = [i / 100 for i in range(0, 101, 10)]
    cost01 = 3  # 设置代价
    cost10 = 2
    theta = np.linspace(0, 1, 10)  # 设置阈值
    draw_cost_curve(output_value, sample, cost01, cost10, theta)  # 绘制代价曲线
