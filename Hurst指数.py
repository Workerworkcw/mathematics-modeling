import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def hurst(ts, if_detail=False):
    # 数据少于20则样本太少
    N = len(ts)
    # 抛出一个自定义的 ValueError（值错误）异常，并给出错误信息 “Time series is too short! input series ought to have at least 20 samples!”。
    if N < 20:
        raise ValueError("Time series is too short! input series ought to have at least 20 samples!")
    # 最大分了max_k个区间，也就是连续两个值作为一个区间
    max_k = int(np.floor(N / 2))
    n_all = []
    RS_all = []
    # k 是子区间长度
    for k in range(5, max_k + 1):
        # 切成K个子序列，将其自动转换成k列的数据
        # 去前N - N % k个数据，把它转换成n行K列
        subset_list = np.array(ts[:N - N % k]).reshape(-1, k).T
        # 累积极差
        # 每个子区间计算离差
        cumsum_list = (subset_list - subset_list.mean(axis=0)).cumsum(axis=0)
        # 每个子区间计算极差
        R = cumsum_list.max(axis=0) - cumsum_list.min(axis=0)
        # 计算各个子区间的重标极插值
        S = (((subset_list - subset_list.mean(axis=0)) ** 2).mean(axis=0)) ** 0.5
        RS = (R / S).mean()
        n_all.append(k)
        RS_all.append(RS)
    # print(k)
    # R_S_all = pd.DataFrame(R_S_all)
    # R_S_all['logN'] = np.log10(R_S_all.n)
    # R_S_all['logRS'] = np.log10(R_S_all.RS)
    # 进行多项式拟合，并且取出斜率作为Hurst指数
    Hurst_exponent = np.polyfit(np.log10(n_all), np.log10(RS_all), 1)[0]
    if if_detail:
        n_all = np.array(n_all)
        RS_all = np.array(RS_all)
        Vn = RS_all / np.sqrt(n_all)
        res = pd.DataFrame([n_all, RS_all, Vn]).T
        res.columns = ['n', 'RS', 'Vn']
        return res, Hurst_exponent
    # plt.plot(R_S_all.logN,R_S_all.logRS)
    else:
        return Hurst_exponent



