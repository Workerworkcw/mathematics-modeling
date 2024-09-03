import numpy as np
import matplotlib.pyplot as plt

# S 易感染者
# I 感染者
# T 时间
# lamda 感染率
# gamma 治愈率
# R 治愈后不会再感染的人群
# E潜伏者
# N 总人数

# population
N = 1e7 + 10 + 5
# simuation Time / Day
T = 170
# susceptiable ratio
s = np.zeros([T])
# exposed ratio
e = np.zeros([T])
# infective ratio
i = np.zeros([T])
# remove ratio
r = np.zeros([T])


# contact rate
lamda = 0.5
# recover rate
gamma = 0.0821

# 潜伏期

# exposed period



sigma = 1 / 4

# initial infective people
i[0] = 10.0 / N
s[0] = 1e7 / N
e[0] = 40.0 / N


for t in range(T-1):
    s[t + 1] = s[t] - lamda * s[t] * i[t]
    e[t + 1] = e[t] + lamda * s[t] * i[t] - sigma * e[t]
    i[t + 1] = i[t] + sigma * e[t] - gamma * i[t]
    r[t + 1] = r[t] + gamma * i[t]


