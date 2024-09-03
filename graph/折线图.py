import matplotlib.pyplot as plt
import numpy as np


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