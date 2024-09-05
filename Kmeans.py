# KMeans聚类分析法，使用轮廓系数法进行分类
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_excel('D:/Python/test/test.xlsx')
# 提取分类依据
X = data[["小区横坐标", "小区纵坐标"]]
# KMeans分类，n_clusters为分类的K值，X为KMeans分类的数据
km = KMeans(n_clusters=25).fit(X)
# labels_分类的标签，就是该坐标对应的类别
data['cluster'] = km.labels_
# 依据类别排序
print(data)
data.sort_values('cluster')
print(data)
#类中心
cluster_centers = km.cluster_centers_
cluster_coordinate = pd.DataFrame(cluster_centers)
x_coordinate = []
y_coordinate = []
for i in range(len(data)):
    for j in range(25):
        if data['cluster'].values[i] == j:
            x_coordinate.append(cluster_coordinate.iloc[j, 0])
            y_coordinate.append(cluster_coordinate.iloc[j, 1])
data['聚类中心横坐标'] = x_coordinate
data['聚类中心纵坐标'] = y_coordinate


# 计算距离
def distance(x, y):
    d = np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
    return d


x1 = data[['小区横坐标', '小区纵坐标']]
x2 = data[['聚类中心横坐标', '聚类中心纵坐标']]
x1_ = np.array(x1)
x2_ = np.array(x2)
# 计算各中心点之间的距离
distance(x1_[1], x2_[1])
# 计算各点到中心的距离
dis = []
for i in range(len(data)):
    dis.append(distance(x1_[i], x2_[i]))
data['距离'] = dis
# 根据分组并进行计数,将小区编号根据cluster分组进行计数
village_num = data.groupby('cluster')['小区编号'].count()
# 组内总人数
peo_num = data.groupby('cluster')['小区人口数（人）'].sum()
# 组内距离和
dis_sum = data.groupby('cluster')['距离'].sum()
# 组内平均值
dis_mean = data.groupby('cluster')['距离'].mean()
# 组内横坐标平均值
area_x = data.groupby('所属区域')['小区横坐标'].mean()
# 组内纵坐标平均值
area_y = data.groupby('所属区域')['小区纵坐标'].mean()

# 坐标平均值统计
area = []
for i in range(9):
    area.append((area_x[i], area_y[i]))

# 组内中心点到其他区域平均坐标距离统计
total = []
for i in range(len(cluster_centers)):
    tmp = []
    for j in range(len(area)):
        # 中心点坐标到区域左边平均值坐标距离计算
        d = distance(cluster_centers[i], area[j])
        tmp.append(d)
    total.append(tmp)


belong_area = []
for i in range(len(total)):
    # 找到total[i]最小值的索引
    ba = total[i].index(min(total[i]))
    belong_area.append(ba)

cluster_coordinate.columns = ['小区横坐标', '小区纵坐标']
cluster_coordinate['管辖范围小区个数'] = village_num.tolist()
cluster_coordinate['管辖范围内人口数'] = peo_num.tolist()
cluster_coordinate['总服务距离'] = dis_sum.tolist()
cluster_coordinate['平均距离'] = dis_mean.tolist()
cluster_coordinate['所属区域'] = belong_area
area_name = area_x.index.tolist()

d_area = {}
for i in range(9):
    d_area[i] = area_name[i]

# 所属行政区统计
area_belong = []
for i in range(len(cluster_coordinate)):
    for j in range(len(area_name)):
        if cluster_coordinate['所属区域'].iloc[i] == j:
            area_belong.append(area_name[j])
cluster_coordinate['所属行政区'] = area_belong

plt.scatter(data["小区横坐标"], data["小区纵坐标"], c=colors[data['cluster']])
plt.scatter(cluster_centers.小区横坐标, cluster_centers.小区纵坐标, linewidth=3, marker='*', s=30, c='r')

# 散点图，根据类别着色
plt.scatter(data["小区横坐标"], data["小区纵坐标"], c=colors[data["cluster"]])
# 中心散点图,颜色标红
plt.scatter(cluster_centers.小区横坐标, cluster_centers.小区纵坐标, linewidths=3, marker='*', s=30, c='r')

# 创建一个新的坐标轴
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("横坐标")
plt.ylabel("纵坐标")
plt.savefig('中心.jpg')
# 散点图 同上
plt.scatter(data["小区横坐标"], data["小区纵坐标"], c=colors[data["cluster"]], s=10, alpha=1)
ax = plt.axes()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('小区坐标.jpg')
plt.scatter(data[["小区横坐标", "小区纵坐标"]], s=100, alpha=1, c=colors[data["cluster"]], figsize=(10, 10))
plt.suptitle("With 2centroids initialized")

#轮廓系数
score = silhouette_score(X, data.cluster)
print(score)
scores = []
for k in range(2, 30):
    labels = KMeans(n_clusters=k).fit(X).labels_
    score = silhouette_score(X, labels)
    scores.append(score)

plt.plot(list(range(2, 30)), scores)
plt.xlabel("Number of Clusters Initialized")
plt.ylabel("Silhouette Score")
