from scipy.optimize import linprog
import pulp

# 目标函数系数（求最小化，这里假设目标函数为 c^T * x）
c = [-1, 4]

# 不等式约束矩阵 A_ub 和向量 b_ub
A_ub = [[-3, 1], [1, 2]]
b_ub = [6, 4]

# 等式约束矩阵 A_eq 和向量 b_eq
A_eq = [[1, -2]]
b_eq = [0]

# 变量的边界
bounds = [(None, None), (-3, None)]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
print(res)

# 创建线性规划问题对象
prob = pulp.LpProblem("example", pulp.LpMaximize)

# 定义变量
x = pulp.LpVariable("x", lowBound=0)
y = pulp.LpVariable("y", lowBound=0)

# 设置目标函数
prob += -x + 4*y
# 添加约束条件
prob += -3*x + y <= 6
prob += x + 2*y <= 4
prob += x - 2*y == 0

# 求解问题
status = prob.solve()
# 输出结果
print("Status:", pulp.LpStatus[status])
print("x=", pulp.value(x))
print("y=", pulp.value(y))



