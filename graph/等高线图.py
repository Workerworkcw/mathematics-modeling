import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# 绘制等高线图
plt.contour(X, Y, Z)

# 添加标题和坐标轴标签
plt.title('Contour Plot')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图形
plt.show()