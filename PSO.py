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


# 随机位置代表的是所需要模拟的参数，在这里面代表的是
# 随机生成位置和速度
position = np.random.randn(50, 4)
velocity = np.random.randn(50, 4)
# 记录个体最佳记录
best_position = position # 个体初始的最佳位置就是初始位置
best_value = np.zeros(50)
# 记录群体最佳位置和最佳值
gbest_position = [0, 0, 0, 0]
gbest_value = 0

# 循环50次
for i in range(50):
    # 50个数据
    for j in range(50):
        position[j] += velocity[j]
        velocity[j] =





