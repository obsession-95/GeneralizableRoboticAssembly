# encoding: utf-8
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体为宋体（SimSun）
# plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class Average_Filter(object):
    def __init__(self, window_size=5):
        # 创建固定长度的双向队列
        self.queue = deque(maxlen=window_size)
        self.average_value = None


    def average_filter(self, ft):
        # 将新读取的六维力数据添加到队列中
        self.queue.append(ft)

        # 如果队列未满，则更新平均值
        # if len(self.queue) < self.queue.maxlen:
        if self.average_value is None:
            self.average_value = ft
        else:
            # axis不设置值，对m*n个数求平均值，返回一个实数
            # axis = 0：压缩行，对各列求均值，返回1*n的矩阵
            # axis = 1: 压缩列，对各行求均值，返回m*1的矩阵
            self.average_value = np.mean(self.queue, axis=0)
        # 如果队列已满，移除队列最前面的值并更新平均值
        # else:
        #     removed_value = self.queue[0]
        #     self.average_value = ((self.average_value * len(self.queue))
        #                           - removed_value + ft) / len(self.queue)
        
        return self.average_value
    

class Kalman_Filter(object):
    def __init__(self, Q, R):
        self.Q = Q
        self.R = R

        self.f_last = [None for _ in range(6)]
        self.f_p_last = [Q for _ in range(6)]

    def kalman_filter(self, ft):
        n = len(ft)
        
        f_mid, f_p_mid = [0 for _ in range(n)], [0 for _ in range(n)]
        f_kg = [0 for _ in range(n)]
        f_now, f_p_now = [0 for _ in range(n)], [0 for _ in range(n)] 

        for i in range(n):
            if self.f_last[i] is None:
                self.f_last[i] = ft[i]

            # 状态预测
            f_mid[i] = self.f_last[i]
            # 协方差预测
            f_p_mid[i] = self.f_p_last[i] + self.Q
            # 卡尔曼增益
            f_kg[i] = f_p_mid[i] / (f_p_mid[i] + self.R)
            # 状态更新
            f_now[i] = f_mid[i] + f_kg[i] * (ft[i] - f_mid[i])
            # 协方差更新
            f_p_now[i] = (1 - f_kg[i]) * f_p_mid[i]
            self.f_p_last[i] = f_p_now[i]
            self.f_last[i] = f_now[i]

        return np.array(f_now)
    

if __name__ == '__main__':
    "均值滤波"
    average_filter = Average_Filter(window_size=30)
    "卡尔曼滤波"
    Q = 0.001  # 过程激励噪声的协方差，状态转移矩阵与实际过程之间的误差
    R = 0.5    # 观测噪声的协方差矩阵，此值一般取绝于测量器件的固有噪声
    kalman_filter = Kalman_Filter(Q=Q, R=R)
    np.random.seed(28)

    ft = [np.array([np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1), 
                    np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1)]) for i in range(1, 500)]  # 六维测量值

    raw_data = []
    average_filter_data = []
    kalman_filter_data = []

    for idx, measurement in enumerate(ft):
        raw_data.append(measurement[0]) # 只记录第一维的数据
        average_filter_data.append(average_filter.average_filter(measurement)[0])
        kalman_filter_data.append(kalman_filter.kalman_filter(measurement)[0])

    # plt.style.use('seaborn')
    config = {
    "font.family":'serif',
    'font.size': 16,
    "font.serif": ['STSong']}
    plt.rcParams.update(config)

    "绘制图表"
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(range(len(raw_data)), raw_data, label='Raw Data', marker='', linewidth=1, color=(0,0.5,1))
    plt.plot(range(len(average_filter_data)), average_filter_data, label='average_filter 均值滤波(window=30)', 
             linestyle='-', marker='', linewidth=1.5, color=(1,0,0))
    plt.plot(range(len(kalman_filter_data)), kalman_filter_data, label='kalman_filter 卡尔曼滤波(Q=0.001, R=0.1)', 
             linestyle='-', marker='', linewidth=1.5, color=(1,0.5,0))
    plt.title('Force Filter 力传感器滤波')
    plt.xlabel('Measurement Index')
    plt.ylabel('Force Value')
    plt.legend()
    plt.grid(True)


    plt.show()

        