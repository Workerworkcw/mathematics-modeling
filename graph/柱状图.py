import matplotlib.pyplot as plt
import numpy as np

# 生成数据
labels = ['A', 'B', 'C', 'D', 'E']
values = [10, 15, 7, 12, 9]

# 绘制柱状图
plt.bar(labels, values)

# 添加标题和坐标轴标签
plt.title('Bar Plot')
plt.xlabel('Categories')
plt.ylabel('Values')

# 显示图形
plt.show()
