import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

N = 8
x = np.linspace(0, 2*np.pi, N, endpoint=False)
height = np.random.randint(3, 15, size=N)

# 画图
# polar: 极坐标
axes = plt.subplot(111, projection='polar')
axes.bar(x=x, height=height, width=0.5, color='b')
plt.show()
