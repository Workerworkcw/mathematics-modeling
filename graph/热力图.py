import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 生成随机数据
data = np.random.rand(10, 10)
cities = np.array(['山东', '黑龙江', '安徽', '江苏', '单县', '曹县', '定陶', '菏泽', '济南', '淄博'])
# 绘制热力图
plt.figure(figsize=(8, 8))
plt.imshow(data, cmap='Blues')
# 修改刻度
plt.xticks(range(len(cities)), cities)
plt.yticks(range(len(data)), data)
# 添加文字
for i in range(len(data)):
    for j in range(len(cities)):
        plt.text(x=i,
                 y=j,
                 s=data[j][i],
                 ha='center',
                 va='center',
                 fontsize=11
                 )


# 颜色条
plt.colorbar()
# 添加标题
plt.title('Heatmap')
# 显示图形
plt.show()

