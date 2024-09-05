import numpy as np
from scipy.optimize import minimize


# 定义目标函数
def objective(x):
    return x[0]**2 + x[1]**2


# 定义不等式约束函数
def constraint_ineq(x):
    return [x[0] + x[1] - 1, -x[0], -x[1]]


# 定义等式约束函数
def constraint_eq(x):
    return [x[0]**2 - x[1]]


# 设置初始值
x0 = np.random.rand(1, 2)[0]
print(x0)

# 定义约束条件字典
con_ineq = {'type': 'ineq', 'fun': constraint_ineq}
con_eq = {'type': 'eq', 'fun': constraint_eq}
constraints = [con_ineq, con_eq]

# 进行优化求解
solution = minimize(objective, x0, constraints=constraints)

print("最优解：", solution.x)
print("最优值：", solution.fun)