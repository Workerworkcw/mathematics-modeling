import math
import string
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import random

# np中的错误处理方式
np.seterr(divide='ignore', invalid='ignore')


# 生成区间[a,b]内的随机数
def random_number(a, b):
    return (b - a) * random.random() + a


# 生成一个矩阵，大小为m*n,并且设置默认零矩阵
def makematrix(m, n, fill=0.0):
    a = []
    for i in range(m):
        # 创建n个[fill] 的列表
        a.append([fill] * n)
    return np.array(a)


# 函数sigmoid(),两个函数都可以作为激活函数
def sigmoid(x):
    return (1 - np.exp(-1 * x)) / (1 + np.exp(-1 * x))


# 函数sigmoid的派生函数
def derived_sigmoid(x):
    return 1 - (np.tanh(x)) ** 2
    # return (2*np.exp((-1)*x)/((1+np.exp(-1*x)**2)))


# 构造三层BP网络架构
class BPNN:
    def __init__(self, num_in, num_hidden, num_out):
        # 输入层，隐藏层，输出层的节点数
        self.num_in = num_in + 1  # 增加一个偏置结点
        self.num_hidden = num_hidden + 1  # 增加一个偏置结点
        self.num_out = num_out

        # 激活神经网络的所有节点（向量）
        # 输入激活
        self.active_in = np.array([-1.0] * self.num_in)
        # 隐藏层激活
        self.active_hidden = np.array([-1.0] * self.num_hidden)
        # 输出激活
        self.active_out = np.array([1.0] * self.num_out)

        # 创建权重矩阵
        # weight,需要修改
        self.wight_in = np.zeros((self.num_in, self.num_hidden), dtype=np.float64)
        self.wight_out = np.zeros((self.num_hidden, self.num_out), dtype=np.float64)

        # 对权值矩阵赋初值
        for i in range(self.num_in):
            for j in range(self.num_hidden):
                self.wight_in[i][j] = random_number(0.1, 0.1)
        for i in range(self.num_hidden):
            for j in range(self.num_out):
                self.wight_out[i][j] = random_number(0.1, 0.1)
        # 偏差
        for j in range(self.num_hidden):
            self.wight_in[0][j] = 0.1
        for j in range(self.num_out):
            self.wight_out[0][j] = 0.1

        # 最后建立动量因子（矩阵）
        self.ci = np.zeros((self.num_in, self.num_hidden), dtype=np.float64)
        self.co = np.zeros((self.num_hidden, self.num_out), dtype=np.float64)

    # 信号正向传播
    def update(self, inputs):
        if len(inputs) != self.num_in - 1:
            raise ValueError('与输入层节点数不符')
        # 数据输入输入层
        self.active_in[1:self.num_in] = inputs

        # 数据在隐藏层的处理
        self.sum_hidden = np.dot(self.wight_in.T, self.active_in.reshape(-1, 1))  # 点乘
        self.active_hidden = sigmoid(self.sum_hidden)  # active_hidden[]是处理完输入数据之后存储，作为输出层的输入数据
        self.active_hidden[0] = -1

        # 数据在输出层的处理
        self.sum_out = np.dot(self.wight_out.T, self.active_hidden)  # 点乘
        self.active_out = sigmoid(self.sum_out)  # 与上同理
        return self.active_out

    # 误差反向传播
    def errorbackpropagate(self, targets, lr, m):  # lr是学习率
        if self.num_out == 1:
            targets = [targets]
        if len(targets) != self.num_out:
            raise ValueError('与输出层节点数不符！')
        # 误差
        error = (1 / 2) * np.dot((targets.reshape(-1, 1) - self.active_out).T,
                                 (targets.reshape(-1, 1) - self.active_out))

        # 输出误差信号
        self.error_out = (targets.reshape(-1, 1) - self.active_out) * derived_sigmoid(self.sum_out)
        # 隐层误差信号
        self.error_hidden = np.dot(self.wight_out, self.error_out) * derived_sigmoid(self.sum_hidden)

        # 更新权值
        # 隐藏
        self.wight_out = self.wight_out + lr * np.dot(self.error_out, self.active_hidden.reshape(1, -1)).T + m * self.co
        self.co = lr * np.dot(self.error_out, self.active_hidden.reshape(1, -1)).T
        # 输入
        self.wight_in = self.wight_in + lr * np.dot(self.error_hidden, self.active_in.reshape(1, -1)).T + m * self.ci
        self.ci = lr * np.dot(self.error_hidden, self.active_in.reshape(1, -1)).T
        return error

    # 测试
    def test(self, patterns):
        for i in patterns:
            print(i[0:self.num_in - 1], '->', self.update(i[0:self.num_in - 1]))
        return self.update(i[0:self.num_in - 1])

    # 权值
    def weights(self):
        print("输入层权重")
        print(self.wight_in)
        print("输出层权重")
        print(self.wight_out)

    def train(self, pattern, itera=100, lr=0.2, m=0.1):
        for i in range(itera):
            error = 0.0
            for j in pattern:
                inputs = j[0:self.num_in - 1]
                targets = j[self.num_in - 1:]
                self.update(inputs)
                error = error + self.errorbackpropagate(targets, lr, m)
            if i % 10 == 0:
                print('########################误差 %-.5f######################第%d次迭代' % (error, i))


# 实例
X = list(np.arange(-1, 1.1, 0.1))
D = [-0.96, -0.577, -0.0729, 0.017, -0.641, -0.66, -0.11, 0.1336, -0.201, -0.434, -0.5, -0.393, -0.1647, 0.0988, 0.3072,
     0.396, 0.3449, 0.1816, -0.0312, -0.2183, -0.3201]
A = X + D
patt = np.array([A] * 2)
# 创建神经网络，21个输入节点，21个隐藏层节点，1个输出层节点
n = BPNN(21, 21, 21)
# 训练神经网络
n.train(patt)
# 测试神经网络
d = n.test(patt)
# 查阅权重值
n.weights()

plt.plot(X, D)
plt.plot(X, d)
plt.show()
