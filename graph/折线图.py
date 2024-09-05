import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 绘制折线图
plt.plot(x, y)

# 添加标题和坐标轴标签
plt.title('Line Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x, y, color='green', linewidth=2, linestyle='--')

# 显示图形
plt.show()

# # 绘制一线
# plt.figure(figsize=(5, 3))
# x = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# y = [20, 40, 35, 55, 42, 80, 50]
# plt.plot(x, y, color='green', linewidth=2, linestyle='--')
# plt.xlabel('星期')
# plt.ylabel("活跃度")
# plt.title("PYthon")
# # 文本
# for a, b in zip(x, y):
#     plt.text(a, b, ha='center', va='bottom', color='green')
#
# # plt.show()

# 绘制多条线
plt.figure(figsize=(5, 3))
# 生成一个包含15个随机数的数组
x = np.random.randint(0, 10, size=15)
plt.plot(x, marker='*', c='r')
plt.plot(x.cumsum(), marker='*', c='r')
plt.show()

