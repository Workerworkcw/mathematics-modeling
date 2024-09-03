import matplotlib.pyplot as plt

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