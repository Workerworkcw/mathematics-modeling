import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 6))

x = np.linspace(0, 10, 100)
# 图1
axes1 = plt.gca()   # 得到当前轴域
axes1.plot(x, np.exp(x), c='r')
axes1.set_xlabel('time')
axes1.set_ylabel('exp(x')
axes1.tick_params(axis='y', labelcolor='r')
# 图2
axes2 = axes1.twinx()   # 和第一个图共享x轴
axes2.plot(x, np.log(x), c='b')

plt.tight_layout()
plt.show()
