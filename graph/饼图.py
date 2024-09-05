import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


x = [10, 20, 30, 40]
# autopct 显示所占比重
plt.pie(x, autopct='%.0f%%')
plt.show()

cities = np.array(['山东', '黑龙江', '安徽', '江苏'])
values = np.random.randint(1, 10, 4)
plt.figure(figsize=(8, 6))
plt.pie(x=values,
        autopct='%.0f%%',
        pctdistance=0.5,    # 百分比文字的位置
        labels=cities,       # 标签
        labeldistance=1,    # 标签的位置
        shadow=True,        # 阴影
        textprops={'fontsize': 10, 'color': 'white'},    # 文字样式
        explode=(0.01, 0.01, 0.01, 0.01),   # 分裂效果
        startangle=90)
plt.show()

# 数据
sizes = [30, 20, 15, 10, 25]
labels = ['A', 'B', 'C', 'D', 'E']
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']

# 绘制饼图
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

# 添加标题
plt.title('Pie Chart')

# 使饼图为正圆形
plt.axis('equal')

# 显示图形
plt.show()

# 甜甜圈
cities = np.array(['山东', '黑龙江', '安徽', '江苏'])
values = np.random.randint(1, 10, 4)
plt.figure(figsize=(8, 6))
plt.pie(x=values,
        autopct='%.0f%%',
        pctdistance=0.5,    # 百分比文字的位置
        labels=cities,       # 标签
        labeldistance=1,    # 标签的位置
        textprops={'fontsize': 10, 'color': 'white'},   # 文字样式
        # 甜甜圈
        wedgeprops={'width': 0.4, 'edgecolor': 'w'}
        )
plt.show()


# 多个圆环
cities1 = np.array(['山东', '黑龙江', '安徽', '江苏'])
values1 = np.random.randint(1, 10, 4)
cities2 = np.array(['单县', '曹县', '定陶', '菏泽'])
values2 = np.random.randint(1, 10, 4)
plt.figure(figsize=(8, 6))
# 第一个饼图
plt.pie(x=values1,
        autopct='%.0f%%',
        pctdistance=0.5,    # 百分比文字的位置
        labels=cities1,       # 标签
        labeldistance=1,    # 标签的位置
        textprops={'fontsize': 10, 'color': 'white'},   # 文字样式
        # 甜甜圈
        wedgeprops={'width': 0.4, 'edgecolor': 'w'}
        )
# 第二个饼图
plt.pie(x=values2,
        autopct='%.0f%%',
        pctdistance=0.5,    # 百分比文字的位置
        textprobs={'fontsize': 10, 'color': 'white'},
        startangle=90,
        # 半径
        radius=0.6
        )
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

