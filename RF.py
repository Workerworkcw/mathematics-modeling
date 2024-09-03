# Python随机森林回归
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
path = 'C:\\Users\\Administrator\\Desktop\\RFRtest.csv' # 数据路径
# 加载数据
rawdata = pd.read_csv(path).set_index('index')
# 数据预览
print(rawdata.head())
# 输入特征
x = rawdata.drop('y_value', axis=1)
# 目标变量
y = rawdata['y_value']
# 训练集和测试集的分割 30%为测试集，则70%为训练集
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=0)
# 随机森林分类器
rfr = RandomForestRegressor(n_estimators=100, random_state=0)
# 使用训练数据集训练随机森林模型
rfr.fit(x_train, y_train)
# 使用分类器预测测试集
y_pred = rfr.predict(x_test)
# 评估回归性能
print('Mean Squared Error:', mean_squared_error(y_test,y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test,y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test,y_pred)))
# 计算特征重要性
importances = rfr.feature_importances_
print("Importances:", importances)
r2 = r2_score(y_test, y_pred)
print("R²:",r2)
# 绘制测试集散点图和斜线
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='grey', linewidth=2)
plt.title(f'Scatter plot of y_test vs. y_pred\nR² = {r2:.2f}')
plt.xlabel('True Values (y_test)')
plt.ylabel('Predictions (y_pred)')
plt.show()
