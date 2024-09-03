import matplotlib.pyplot as plt
import numpy as np

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