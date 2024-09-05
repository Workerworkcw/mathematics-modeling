import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(-np.pi, np.pi, 30)
y = np.sin(x)

plt.figure(figsize=(8, 6))
ax1 = plt.subplot(2, 2, 1)
ax1.plot(x, y)

ax2 = plt.subplot(2, 2, 2)
ax2.plot(x, y)

ax3 = plt.subplot(2, 2, 3)
ax3.plot(x, y)

ax3 = plt.subplot(2, 2, 4)
ax3.plot(x, y)

plt.show()

ax1 = plt.subplot(2, 2, 1)
ax1.plot(x, y)

ax2 = plt.subplot(2, 2, 2)
ax2.plot(x, y)

ax3 = plt.subplot(2, 1, 2)
ax3.plot(x, y)
plt.show()


# 使用subplots函数
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
# 3行3列
fig, ax = plt.subplots(3, 3)
ax1, ax2, ax3 = ax
ax11, ax12, ax13 = ax1
ax21, ax22, ax23 = ax2
ax31, ax32, ax33 = ax3
# fig设置画布大小
fig.set_figwidth(8)
fig.set_figheight(6)
ax11.plot(x, y)
ax12.plot(x, np.cos(x))
ax13.plot(x, np.tan(x))

ax21.plot(x, y)
ax22.plot(x, np.cos(x))
ax23.plot(x, np.tan(x))

ax31.plot(x, y)
ax32.plot(x, np.cos(x))
ax33.plot(x, np.tan(x))

plt.tight_layout()

plt.show()





