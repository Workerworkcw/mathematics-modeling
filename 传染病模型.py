import numpy as np
import matplotlib.pyplot as plt

# S 易感染者
# I 感染者
# T 时间
# lamda 感染率
# gamma 治愈率
# R 治愈后不会再感染的人群
# E 潜伏者
# N 总人数

# population        总人口
N = 1e7 + 10 + 5
# simuation Time / Day 仿真时间
T = 170
# susceptiable ratio    感染比例
s = np.zeros([T])
# exposed ratio     接触比例
e = np.zeros([T])
# infective ratio   感染比例
i = np.zeros([T])
# remove ratio      死亡比例
r = np.zeros([T])


# contact rate      感染率
lamda = 0.5
# recover rate      痊愈率
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


fig, ax = plt.subplots(figsize=(10,6))
ax.plot(s, c='b', lw=2, label='S')
ax.plot(i, c='r', lw=2, label='I')
ax.plot(r, c='g', lw=2, label='R')
ax.set_xlabel('Day',fontsize=20)
ax.set_ylabel('Infective Ratio', fontsize=20)
ax.grid(1)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend();



