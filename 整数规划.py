import pulp


# 创建整数规划问对象
prob = pulp.LpProblem("example", pulp.LpMaximize)

# 定义整数变量
x = pulp.LpVariable("x", lowBound=0, cat="binary")
y = pulp.LpVariable("y", lowBound=0, cat="binary")

# 设置目标函数
prob += 3*x + 2*y

# 添加约束条件
prob += x + y <= 5
prob += x*2 + y*3 <= 100

# 求解问题
status = prob.solve()
print("Status:", pulp.LpStatus[status])
print("x=", pulp.value(x))
print("y=", pulp.value(y))
