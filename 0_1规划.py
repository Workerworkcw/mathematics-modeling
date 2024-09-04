import pulp

# 创建 0-1 规划问题对象
prob = pulp.LpProblem("example", pulp.LpMaximize)

# 定义 0-1 变量
x = pulp.LpVariable("x", cat='Binary')
y = pulp.LpVariable("y", cat='Binary')

# 设置目标函数
prob += 3*x + 2*y

# 添加约束条件
prob += x + y <= 1
prob += 2*x + 3*y >= 4

# 求解问题
status = prob.solve()

# 输出结果
print("Status:", pulp.LpStatus[status])
print("x =", pulp.value(x))
print("y =", pulp.value(y))