import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import lagrange


# 牛顿插值
def newton_interpolation(x, y, x_interp):
    n = len(x)
    # 计算差商表
    divided_differences = np.zeros((n, n))
    divided_differences[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            divided_differences[i][j] = (divided_differences[i + 1][j - 1] - divided_differences[i][j - 1]) / (x[i + j] - x[i])
    # 计算插值结果
    result = divided_differences[0][0]
    term = 1
    for i in range(1, n):
        term *= (x_interp - x[i - 1])
        result += divided_differences[0][i] * term
    return result

# 已知数据点
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])

# 进行插值计算
x_interp = 2.5
y_interp = newton_interpolation(x, y, x_interp)
print(f"在 x = {x_interp} 处的插值结果为：{y_interp}")

# 创建拉格朗日插值函数
f = lagrange(x, y)

# 进行插值计算
x_interp = 2.5
y_interp = f(x_interp)
print(f"在 x = {x_interp} 处的插值结果为：{y_interp}")


# 拉依达（PauTa）准则，也称为 3σ 准则
def detect_outliers(data):
    mean = np.mean(data)
    std = np.std(data)
    print(mean, std)
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    print(lower_bound, upper_bound)
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return outliers


data = [1, 2, 3, 4, 5, 100, 6, 7, 8, 9]
outliers = detect_outliers(data)
print(f"异常值为：{outliers}")
