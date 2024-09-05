import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 图形嵌套 add_subplot()函数
fig = plt.figure(figsize=(8, 6))
# 子图1
axes1 = fig.add_subplot(1, 1, 1)
axes1.plot([0, 1], [0, 1])
# 子图2：嵌套图
axes2 = fig.add_subplot(2, 2, 1,facecolor='pink')
axes2.plot([1, 3])
plt.show()

# 使用axes()函数
# 使用add_axes()函数
fig = plt.figure(figsize=(8, 6))
# 图1
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
# 嵌套图1
# axes  [left, bottom. width, height]
axes1 = plt.axes([0.5, 0.5, 0.5, 0.5])
axes1.plot(x, y, color='red')
# 嵌套图2
axes2 = fig.add_axes([0.5, 0.5, 0.5, 0.5])
axes2.plot(y, x, color='blue')
plt.show()



