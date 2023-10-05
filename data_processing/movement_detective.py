import numpy as np
import matplotlib.pyplot as plt
'''
一般来说实现动作的提取有三种方法
1.自动标记：在测数据的时候你规定好什么时候进行一次动作，比如说两秒为周期？ 但是事实上不可能如此的规律，导致动作提取是紊乱的
2.手动标记：在采集csi的同时加一个摄像头，根据摄像头的拍摄结果手动标记时间戳，这样子准确但是工作量极大
3.基于滑动方差标记：记录下稳态的csi数据，随后窗口内出现较大的方差变动则判定为有动作
'''
#根据csi数据返回滑动方差值
def moving_variance(csi_data, subcarrier_idx, k=10): #k为当前点向前考虑的点数
    # 提取所有时间戳、天线的给定子载波的幅度值
    signal = csi_data[:,0,0,subcarrier_idx]
    
    #plt.plot(range(len(signal)),signal,label = 'csi',c = 'r')
    #plt.show()
    
    variances = []
    for n in range(len(signal)):
        if n < k:
            variances.append(0)  # 或者可以使用现有点计算方差
            continue
        M_n_k_to_n = signal[n-k:n]  # 包括当前值的前k个信号
        M_avg = sum(M_n_k_to_n) / k
        variance = sum([(M_i - M_avg)**2 for M_i in M_n_k_to_n]) / k
        variances.append(variance)
    
    return variances

#根据信号的移动方差计算活动的开始和结束
def detect_activity(variances):
    threshold_low = 1
    threshold_high = 10
    activities = [1 if threshold_low < s2 < threshold_high else 0 for s2 in variances]
    return activities

#绘制原始的csi数据和提取的动作区间 主要是为了评估效果
def plot_csi_with_activity(csi_data, subcarrier_idx, activities, grace_period=400):
    """
    绘制CSI数据，并根据活动列表标注动作区间。

    参数:
    - csi_data: ndarray, 预处理的CSI数据
    - subcarrier_idx: int, 子载波索引
    - activities: list, 动作标注，1 表示有动作，0 表示没有
    - grace_period: int, 允许的连续无动作的最大长度，超过这个值才断开区间
    """
    # 绘制CSI数据
    plt.plot(range(len(csi_data)), csi_data[:, 0, 0, subcarrier_idx], label='CSI DATA', color='g')
    
    # 标注动作
    start_activity = None
    no_activity_count = 0
    for i, activity in enumerate(activities):
        if activity == 1 and start_activity is None:
            start_activity = i
        elif activity == 0:
            if start_activity is not None:
                no_activity_count += 1
                if no_activity_count > grace_period:
                    plt.axvspan(start_activity, i - no_activity_count, color='red', alpha=0.5)
                    start_activity = None
                    no_activity_count = 0
        else:
            no_activity_count = 0

    # 如果在最后还有持续的动作，关闭区间
    if start_activity is not None:
        plt.axvspan(start_activity, len(activities), color='red', alpha=0.5)
    
    plt.title("CSI Data with Detected Activity")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()



