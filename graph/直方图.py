import matplotlib.pyplot as plt
import numpy as np

# 生成随机数据
data = np.random.randn(1000)

# 绘制直方图
plt.hist(data, bins=30)

# 添加标题和坐标轴标签
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 显示图形
plt.show()