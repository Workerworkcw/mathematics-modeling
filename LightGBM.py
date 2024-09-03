import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from IPython.core.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# 对于 xlsx 的文件
df = pd.read_excel(r'D:\MODEL\model2\huiguiall.xlsx')
df.head()
X = df[['土壤蒸发量(mm)', '降水天数', '平均能见度(km)', '平均最大持续风速(knots)', '平均露点温度(℃)']]
Y = df[['10cm 湿度(kg/m2)']]
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
# model = lgb.LGBMRegressor(max_depth=6, learning_rate=0.14) # 回归
# model = lgb.LGBMClassifier() # 分类
model = lgb.LGBMRegressor() # 回归
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# # 写入，方便比较
# a = pd.DataFrame() # 创建一个空的 DataFrame
# a['预测值'] = list(y_pred)
# a['实际值'] = list(y_test)
# 模型误差（回归）
print('平均绝对误差 MAE： ', mean_absolute_error(y_test, y_pred))
print('均方误差 MSE： ', mean_squared_error(y_test, y_pred))
# print('均方误差对数 MSLE ： ', mean_squared_log_error(y_pred, y_test))
print('中位绝对误差： ', median_absolute_error(y_test, y_pred))
print('可决系数 R^2 ： ', r2_score(y_test, y_pred))
# 模型准确度评分（分类）
# score = accuracy_score(y_test， y_pred)
# model.score(X_test, y_test)
# 模型 ROC 曲线（分类）
# y_pred_proba = model.predict_proba(X_test)
# fpr, tpr, thres = roc_curve(y_test, y_pred_proba[:, 1])
# plt.plot(fpr, tpr)
# plt.show()
# 模型 ROC 曲线的 AUC 值（分类）
# score = roc_auc_score(y_test.values, y_pred_proba[:, 1])
# 特征重要性排序
features = X.columns # 获取特征名称
importances = model.feature_importances_ # 获取特征重要性
# 通过二维表格形式显示
importances_df = pd.DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
importances_df.sort_values('特征重要性', ascending=False)
# 模型参数调优
# parameters = {'num_leaves': [10, 15, 31], 'n_estimators': [10, 20, 30], 'learning_rate':[0.05, 0.1, 0.2]}
parameters = {'max_depth': [2, 3, 4, 6, 8, 10, 12, 15, 17, 20], 'learning_rate': [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.12, 0.14, 0.16, 0.24, 0.26]}
grid_search = GridSearchCV(model, parameters, cv=5) # 5 折交叉验证
grid_search.fit(X_train, y_train) # 传入数据
print('最优参数： ', grid_search.best_params_)
print('最优得分： ', grid_search.best_score_)
# print('不同参数情况下交叉验证的结果： ', grid_search.cv_results_)
# 结果： {'learning_rate': 0.1, 'n_estimators': 20, 'num_leaves': 15}
"""画出拟合曲线"""
fig = plt.figure()
plt.rcParams['font.sans-serif'] = 'default' # 中文黑体为'SimHei'
plt.rcParams['axes.unicode_minus'] = False # 中文时，正常显示负号
font_dict = dict(fontsize=13,color='k',family='default', weight='light',style='italic',)  # 中文黑体为'SimHei'
plt.title('Curve Fitting', fontdict=font_dict)
plt.xlabel('Date', fontdict=font_dict)
plt.ylabel('Qualified Rate', fontdict=font_dict)
plt.scatter(X_test, y_test, label='Label', linestyle='-', color='r', marker='*', linewidth=2.5)
plt.scatter(X_test, y_pred, label='Prediction', linestyle='--', color='b', marker='*',linewidth=2.5)
plt.legend()
plt.show()
fig.savefig('拟合曲线.png')
