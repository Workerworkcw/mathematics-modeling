import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 生成随机数据
x = np.random.randint(0, 10, 100)

# 统计每个数出现的次数
print(pd.Series(x).value_counts())

# 绘制直方图
# bins组数
plt.hist(x,bins=10)
# 按0-3,3-6,6-9,9-10分组
# 颜色、透明度、边框
# density=True 概率分布
plt.hist(x, bins=[0, 3, 6, 9, 10], facecolor='g', alpha=0.75, density=True)
plt.show()

# x轴调整
plt.xticks(np.arange(10))

# 添加标题和坐标轴标签
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 显示图形
plt.show()