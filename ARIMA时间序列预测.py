import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from matplotlib import ticker
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller as ADF



plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成时间等差数列
t = np.linspace(0, 100, 100)
# 随机生成100个数据
value = np.random.randn(100)

# 划分训练集和测试集,前70个训练，后30个测试
Chinese_bank = pd.DataFrame({'t': t, 'v': value})

train = Chinese_bank.loc['0':'69']
test = Chinese_bank.loc['70':'99']
# 查看训练集
plt.plot(train, 'b-', label='train')
plt.xticks(rotation=90)
plt.show()
# 差分法
Chinese_bank['diff1'] = Chinese_bank['v'].diff(1)     # 一阶差分
Chinese_bank['diff2'] = Chinese_bank['diff1'].diff(1)       # 二阶差分

# 作图比较
ax0 = plt.subplot(311)
ax0.plot(t_train, Chinese_bank['t'])
ax0.set_title('原始数据')
ax1 = plt.subplot(312)
ax1.plot(t_train, Chinese_bank['diff1'])
ax1.set_title('一阶差分')
ax2 = plt.subplot(313)
ax2.plot(t_train, Chinese_bank['diff2'])
ax2.set_title('二阶差分')
plt.show()

# 计算原始序列、一阶差分序列、二阶差分序列的单位根检验结果
Chinese_bank['diff1'] = Chinese_bank['diff1'].fillna(0)
Chinese_bank['diff2'] = Chinese_bank['diff2'].fillna(0)

timeseries = ADF(Chinese_bank['v'].tolist())
timeseries_diff1 = ADF(Chinese_bank['diff1'].tolist())
timeseries_diff2 = ADF(Chinese_bank['diff2'].tolist())
print(timeseries)
print(timeseries_diff1)
print(timeseries_diff2)

# 参数确定
plt.figure(figsize=(10, 6))
ax1 = plt.subplot(211)
fig = sm.graphics.tsa.plot_acf(train, lags=20, ax=ax1)
ax1.xaxis.set_ticks_position('bottom')

plt.show()

ax2 = plt.subplot(212)
fig = sm.graphics.tsa.plot_acf(train, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
plt.show()

# 确定pdq的取值范围
p_min = 0
d_min = 0
q_min = 0
p_max = 5
d_max = 0
q_max = 5

# Initialize a Dataframe to store the results, BIC
result_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                          columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])

for p, d, q in itertools.product(range(p_min, p_max + 1),
                                 range(d_min, d_max + 1),
                                 range(q_min, q_max + 1)):
    if p == 0 and d == 0 and q == 0:
        result_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue
    try:
        model = sm.tsa.ARIMA(train, order=(p, d, q))
        result = model.fit()
        result_bic['AR{}'.format(p), 'MA{}'.format(q)] = result.bic
    except Exception as e:
        continue

print(result_bic)

# 得到结果进行浮点类型转换
result_bic = result_bic[result_bic.columns].astype('float64')

# 绘制热力图
fig, ax = plt.subplots(figsize=(10, 6))
ax = sns.heatmap(result_bic,
                 mask=result_bic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f',
                 cmap="Purples")
ax.set_title('BIC')
plt.show()

# 利用模型获取pq的最优值
train_results = sm.tsa.arima_order_select_ic(train,
                                             ic=['aic', 'bic'],
                                             trend='n',
                                             max_ar=8,
                                             max_ma=8)
print('AIC', train_results.aic_min_order)
print('BIC', train_results.bic_min_order)

# 模型检验
p = 1
d = 0
q = 0
model = sm.tsa.ARIMA(train, order=(p, d, q))
result = model.fit()
resid = result.resid    # 获取残差

# 绘制
# 查看测试集的时间序列与数据（只包含测试集）
fig, ax = plt.subplots(figsize=(10, 6))
ax = sm.graphics.tsa.plot_acf(train['t'], lags=20, ax=ax)
plt.show()

# 模型预测
predict_sunspots = result.predict(dynamic=False)
print(predict_sunspots)
plt.figure(figsize=(10, 6))
plt.plot(train)
plt.xticks(rotation=45)
plt.plot(predict_sunspots)
plt.show()

