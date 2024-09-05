import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 简单柱状图
x = ['语文', '数学', '英语', 'Python', '物理']
y = [20, 10, 40, 60, 10]

plt.figure(figsize=(5, 3))
plt.bar(x, y)
plt.show()

x = np.random.randint(50, 90, size=10)
y = np.array([i for i in range(2014, 2024)])
# width 柱的宽度
plt.bar(x, y, width=0.5)
# 在每个柱子上显示数值
for a, b in zip(x, y):
    plt.text(x=a,
             y=b,
             s='{:.1f}万'.format(b),  # 单位显示
             ha='center',   # 字体居中
             fontsize='9'   # 字体大小
             )
plt.show()


# 一次绘制多个柱状图
x = np.array([i for i in range(2014, 2024)])
y1 = np.random.randint(10, 100, size=10)
y2 = np.random.randint(10, 100, size=10)
y3 = np.random.randint(10, 100, size=10)

plt.figure(figsize=(5, 3))
plt.title('年销售额')
plt.xlabel('年份')
plt.ylabel('销售额')
plt.bar(x, y1)
plt.bar(x, y2)
plt.show()


# 簇状柱形图
# 对上图进行微调
# 让三条柱子并排显示
w = 0.5
plt.bar(x-w, y1, width=w, label='随机1')
plt.bar(x, y2, width=w, label='随机2')
plt.bar(x+w, y3, width=w, label='随机3')
plt.show()

# 堆叠柱状图
plt.bar(x-w, y1,  label='随机1')
plt.bar(x, y2,  label='随机2')
plt.bar(x+w, y3,  label='随机3')
plt.legend()
plt.show()

# 条形图
plt.barh(x, y1)

