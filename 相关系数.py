import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x1 = np.random.randn(100)
x2 = np.random.randn(100)

plt.scatter(x1,x2)
plt.show()
# 皮尔逊系数
print(stats.pearsonr(x1, x2))
# 斯皮尔曼系数
print(stats.spearmanr(x1, x2))
# 肯德尔相关系数
print(stats.kendalltau(x1, x2))

