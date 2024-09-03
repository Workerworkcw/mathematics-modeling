import numpy as np
from scipy.optimize import minimize


# 定义目标函数
def objective(x):
    return np.sum(x**2)


# 定义线性约束条件
A_eq = np.array([[1, 1]])
b_eq = np.array([2])

# 定义变量的上下界
x0 = np.array([0, 0])
bounds = [(0, None), (0, None)]

# 使用 minimize 函数求解
res = minimize(objective, x0,
               method='SLSQP',
               bounds=bounds,
               constraints={'type': 'eq', 'fun': lambda x: A_eq.dot(x) - b_eq})

print(res.x)
