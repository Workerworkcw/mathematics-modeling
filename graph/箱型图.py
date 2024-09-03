import matplotlib.pyplot as plt
import numpy as np

# 生成随机数据
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

# 绘制箱型图
plt.boxplot(data)

# 添加标题和坐标轴标签
plt.title('Box Plot')
plt.xlabel('Data Sets')
plt.ylabel('Values')

# 显示图形
plt.show()