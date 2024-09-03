import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# w惯性权重，控制速度粒子的保留程度，影响算法的全局修改能力
# c1 c2 加速系数，分别代表个体学习因子和社会学习因子，控制粒子向个体最优和全局最优靠拢的程度
# r1 r2 是[0, 1]区间内的随机数，为算法增加随机性
# pbesti 粒子i目前找到的最佳位置
# gbest 粒子群目前找到的最佳位

w = 0.5
c1 = 2
c2 = 2
xmin = 0
xmax = 1








