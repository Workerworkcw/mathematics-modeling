import copy
import time
import random
import numpy as np
import pandas as pd
import os
import math
import folium


class ALNSSearch():
    # 计算TSP总距离
    # 计算TSP总距离
    @staticmethod
    # static_method被定义为静态方法,不依赖于实例即可调用
    def dis_cal(path, dist_mat):
        distance = 0
        for i in range(len(path) - 1):
            distance += dist_mat[path[i]][path[i + 1]]
        distance += dist_mat[path[-1]][path[0]]
        return distance

    # 随机删除N个城市
    @staticmethod
    def random_destroy(x, destroy_city_cnt):
        new_x = copy.deepcopy(x)
        removed_cities = []

        # 随机选择N个城市，并删除
        removed_index = random.sample(range(0, len(x)), destroy_city_cnt)
        for i in removed_index:
            removed_cities.append(new_x[i])
            x.remove(new_x[i])
        return removed_cities

    # 删除距离最大的N个城市
    @staticmethod
    def max_n_destroy(x, destroy_city_cnt):
        new_x = copy.deepcopy(x)
        removed_cities = []

        # 计算顺序距离并排序
        dis = []
        for i in range(len(new_x) - 1):
            dis.append(dist_mat[new_x[i]][new_x[i + 1]])
        dis.append(dist_mat[new_x[-1]][new_x[0]])
        # 对其进行排序，并获取排序后的索引
        sorted_index = np.argsort(np.array(dis))

        # 删除最大的N个城市
        for i in range(destroy_city_cnt):
            removed_cities.append(new_x[sorted_index[-1 - i]])
            x.remove(new_x[sorted_index[-1 - i]])

        return removed_cities

    # 随机删除连续的N个城市
    @staticmethod
    def continue_n_destroy(x, destroy_city_cnt):

        new_x = copy.deepcopy(x)
        removed_cities = []

        # 随机选择N个城市，并删除
        removed_index = random.sample(range(0, len(x) - destroy_city_cnt), 1)[0]
        for i in range(removed_index + destroy_city_cnt, removed_index, -1):
            removed_cities.append(new_x[i])
            x.remove(new_x[i])
        return removed_cities

    # destroy操作
    def destroy(self, flag, x, destroy_city_cnt):
        # 三个destroy算子，第一个是随机删除N个城市，第二个是删除距离最大的N个城市，第三个是随机删除连续的N个城市
        removed_cities = []
        if flag == 0:
            # 随机删除N个城市
            removed_cities = self.random_destroy(x, destroy_city_cnt)
        elif flag == 1:
            # 删除距离最大的N个城市
            removed_cities = self.max_n_destroy(x, destroy_city_cnt)
        elif flag == 2:
            # 随机删除连续的N个城市
            removed_cities = self.continue_n_destroy(x, destroy_city_cnt)

        return removed_cities

    # 随机插入
    @staticmethod
    def random_insert(x, removed_cities):
        insert_index = random.sample(range(0, len(x)), len(removed_cities))
        for i in range(len(insert_index)):
            x.insert(insert_index[i], removed_cities[i])

    # 贪心插入
    def greedy_insert(self, x, removed_cities):
        dis = float('inf')
        insert_index = -1

        for i in range(len(removed_cities)):
            # 寻找插入后的最小总距离
            for j in range(len(x) + 1):
                new_x = copy.deepcopy(x)
                new_x.insert(j, removed_cities[i])
                if self.dis_cal(new_x, dist_mat) < dis:
                    dis = self.dis_cal(new_x, dist_mat)
                    insert_index = j

            # 最小位置处插入
            x.insert(insert_index, removed_cities[i])
            dis = float('inf')

    # repair操作
    def repair(self, flag, x, removed_cities):
        # 两个repair算子，第一个是随机插入，第二个贪心插入
        if flag == 0:
            self.random_insert(x, removed_cities)
        elif flag == 1:
            self.greedy_insert(x, removed_cities)

    # 选择destroy算子
    def select_and_destroy(self, destroy_w, x, destroy_city_cnt):
        # 轮盘赌逻辑选择算子
        prob = destroy_w / np.array(destroy_w).sum()
        seq = [i for i in range(len(destroy_w))]
        destroy_operator = np.random.choice(seq, size=1, p=prob)[0]
        # destroy操作
        removed_cities = self.destroy(destroy_operator, x, destroy_city_cnt)

        return removed_cities, destroy_operator

    # 选择repair算子
    def select_and_repair(self, repair_w, x, removed_cities):
        # # 轮盘赌逻辑选择算子
        prob = repair_w / np.array(repair_w).sum()
        seq = [i for i in range(len(repair_w))]
        repair_operator = np.random.choice(seq, size=1, p=prob)[0]
        # repair操作
        self.repair(repair_operator, x, removed_cities)

        return repair_operator

    # ALNS主程序
    def calc_by_alns(self, dist_mat):
        # 模拟退火温度
        T = 100
        # 降温速度
        a = 0.97

        # destroy的城市数量
        destroy_city_cnt = int(len(dist_mat) * 0.1)
        # destroy算子的初始权重
        destroy_w = [1, 1, 1]
        # repair算子的初始权重
        repair_w = [1, 1]
        # destroy算子的使用次数
        destroy_cnt = [0, 0, 0]
        # repair算子的使用次数
        repair_cnt = [0, 0]
        # destroy算子的初始得分
        destroy_score = [1, 1, 1]
        # repair算子的初始得分
        repair_score = [1, 1]
        # destroy和repair的挥发系数
        lambda_rate = 0.5

        # 当前解，第一代，贪心策略生成
        removed_cities = [i for i in range(dist_mat.shape[0])]
        x = []
        self.repair(1, x, removed_cities)

        # 历史最优解，第一代和当前解相同，注意是深拷贝，此后有变化不影响x，也不会因x的变化而被影响
        history_best_x = copy.deepcopy(x)

        # 迭代
        cur_iter = 0
        max_iter = 100
        print(
            'cur_iter: {}, best_f: {}, best_x: {}'.format(cur_iter, self.dis_cal(history_best_x, dist_mat),
                                                          history_best_x))

        while cur_iter < max_iter:

            # 生成测试解，即伪代码中的x^t
            test_x = copy.deepcopy(x)

            # destroy算子
            remove, destroy_operator_index = self.select_and_destroy(destroy_w, test_x, destroy_city_cnt)
            destroy_cnt[destroy_operator_index] += 1

            # repair算子
            repair_operator_index = self.select_and_repair(repair_w, test_x, remove)
            repair_cnt[repair_operator_index] += 1

            if self.dis_cal(test_x, dist_mat) <= self.dis_cal(x, dist_mat):
                # 测试解更优，更新当前解
                x = copy.deepcopy(test_x)
                if self.dis_cal(test_x, dist_mat) <= self.dis_cal(history_best_x, dist_mat):
                    # 测试解为历史最优解，更新历史最优解，并设置最高的算子得分
                    history_best_x = copy.deepcopy(test_x)
                    destroy_score[destroy_operator_index] = 1.5
                    repair_score[repair_operator_index] = 1.5
                else:
                    # 测试解不是历史最优解，但优于当前解，设置第二高的算子得分
                    destroy_score[destroy_operator_index] = 1.2
                    repair_score[repair_operator_index] = 1.2
            else:
                if np.random.random() < np.exp((self.dis_cal(x, dist_mat) - self.dis_cal(test_x, dist_mat)) / T):
                    # 当前解优于测试解，但满足模拟退火逻辑，依然更新当前解，设置第三高的算子得分
                    x = copy.deepcopy(test_x)
                    destroy_score[destroy_operator_index] = 0.8
                    repair_score[repair_operator_index] = 0.8
                else:
                    # 当前解优于测试解，也不满足模拟退火逻辑，不更新当前解，设置最低的算子得分
                    destroy_score[destroy_operator_index] = 0.5
                    repair_score[repair_operator_index] = 0.5

            # 更新destroy算子的权重
            destroy_w[destroy_operator_index] = \
                destroy_w[destroy_operator_index] * lambda_rate + \
                (1 - lambda_rate) * destroy_score[destroy_operator_index] / destroy_cnt[destroy_operator_index]
            # 更新repair算子的权重
            repair_w[repair_operator_index] = \
                repair_w[repair_operator_index] * lambda_rate + \
                (1 - lambda_rate) * repair_score[repair_operator_index] / repair_cnt[repair_operator_index]
            # 降低温度
            T = a * T

            # 结束一轮迭代，重置模拟退火初始温度
            cur_iter += 1
            print(
                'cur_iter: {}, best_f: {}, best_x: {}'.format(cur_iter, self.dis_cal(history_best_x, dist_mat),
                                                              history_best_x))

        # 打印ALNS得到的最优解
        print(history_best_x)
        print(self.dis_cal(history_best_x, dist_mat))
        return history_best_x


if __name__ == '__main__':
    original_cities = [['西宁', 101.74, 36.56],
                       ['兰州', 103.73, 36.03],
                       ['银川', 106.27, 38.47],
                       ['西安', 108.95, 34.27],
                       ['郑州', 113.65, 34.76],
                       ['济南', 117, 36.65],
                       ['石家庄', 114.48, 38.03],
                       ['太原', 112.53, 37.87],
                       ['呼和浩特', 111.65, 40.82],
                       ['北京', 116.407526, 39.90403],
                       ['天津', 117.200983, 39.084158],
                       ['沈阳', 123.38, 41.8],
                       ['长春', 125.35, 43.88],
                       ['哈尔滨', 126.63, 45.75],
                       ['上海', 121.473701, 31.230416],
                       ['杭州', 120.19, 30.26],
                       ['南京', 118.78, 32.04],
                       ['合肥', 117.27, 31.86],
                       ['武汉', 114.31, 30.52],
                       ['长沙', 113, 28.21],
                       ['南昌', 115.89, 28.68],
                       ['福州', 119.3, 26.08],
                       ['台北', 121.3, 25.03],
                       ['香港', 114.173355, 22.320048],
                       ['澳门', 113.54909, 22.198951],
                       ['广州', 113.23, 23.16],
                       ['海口', 110.35, 20.02],
                       ['南宁', 108.33, 22.84],
                       ['贵阳', 106.71, 26.57],
                       ['重庆', 106.551556, 29.563009],
                       ['成都', 104.06, 30.67],
                       ['昆明', 102.73, 25.04],
                       ['拉萨', 91.11, 29.97],
                       ['乌鲁木齐', 87.68, 43.77]]
    original_cities = pd.DataFrame(original_cities, columns=['城市', '经度', '纬度'])
    D = original_cities[['经度', '纬度']].values * math.pi / 180
    city_cnt = len(original_cities)
    dist_mat = np.zeros((city_cnt, city_cnt))
    for i in range(city_cnt):
        for j in range(city_cnt):
            if i == j:
                # 相同城市不允许访问
                dist_mat[i][j] = 1000000
            else:
                # 单位：km
                dist_mat[i][j] = 6378.14 * math.acos(
                    math.cos(D[i][1]) * math.cos(D[j][1]) * math.cos(D[i][0] - D[j][0]) +
                    math.sin(D[i][1]) * math.sin(D[j][1]))

    # ALNS求解TSP
    time0 = time.time()
    alns = ALNSSearch()
    history_best_x = alns.calc_by_alns(dist_mat)
    print(city_cnt)
    print('使用ALNS求解TSP，耗时: {} s'.format(time.time() - time0))