import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 生成随机数据
data = np.random.rand(10, 10)

# 绘制热力图
sns.heatmap(data)

# 添加标题
plt.title('Heatmap')

# 显示图形
plt.show()