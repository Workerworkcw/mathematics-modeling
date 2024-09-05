import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成随机数据
x = np.random.randn(100)
y = np.random.randn(100)
# 绘制散点图
plt.scatter(x, y)
# 添加标题和坐标轴标签
plt.title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(x, y, c='r', marker='o', s=50)
# 显示图形
plt.show()

# 气泡图
plt.figure(figsize=(5, 3))
data = np.random.randn(100, 2)  # 100行，2列 符合正态分布
# 给个随机尺寸
s = np.random.randint(50, 200, size=100)
# s随机给一些大小
plt.scatter(data[:, 0], data[:, 1], s=s)


# 六边形图
plt.figure(figsize=(5, 3))
# gridsize 点的大小，点的形状变为六边形
plt.hexbin(x, y, gridsize=20, cmap='rainbow')
plt.title('我乱编的数据')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()




