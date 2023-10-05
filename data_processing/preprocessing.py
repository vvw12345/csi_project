import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt

'''
行去中心化，Z-core标准化，min-max标准化
注意二者的区别
行去中心化让每行的变量减去这行均值，常用于PCA的前置步骤，图像处理等
Z-core标准化指对整个矩阵计算均值，减去均值并除以标准差，常用于机器学习的前置步骤，目的是为了让每个特征的权重是相同的
min-max标准化缩小范围，常用于对小范围敏感的函数，比如sigmoid函数
'''
#对csi数据的行向量进行去中心化处理
#消除信号的静态成分
def centerize_csi(csi_data):
    # 计算每个行向量的均值
    row_means = np.mean(csi_data, axis=1, keepdims=True)

    # 去中心化处理
    centered_csi_data = csi_data - row_means

    return centered_csi_data

#Z-core标准化
def z_core_standardization(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    
    standardized_data = (data - mean) / std_dev
    return standardized_data


#min-max标准化 new_min指标准化后数据的最小值 new_max指标准化后数据的最大值
#公式 ： x = (x - min) / (max - min)
def min_max_standardization(data, new_min=0, new_max=1):
    old_min = np.min(data)
    old_max = np.max(data)
    
    standardized_data = (data - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    return standardized_data



'''
信号直流分量去除 一般是定位到频谱index = 0的位置
'''
#对信号先进行FFT变换获取频谱图，然后去除信号直流分量，再进行IFFT变换把信号还原为时域图
#data (numpy.ndarray): 输入信号数据，要求格式为 (N, M, P, L)，其中 N 表示包数，M 表示发射天线数，P 表示接收天线数，L 表示子载波数。
#sampling_frequency指信号采样频率，此处设置默认值为1000
#对于FFT变换得到的频率，是关于y轴对称的，因此直流分量一般来说是在0出现的，因此如果需要去除分量只需要找到0的位置，然后置0即可
def remove_dc_component(data, sampling_frequency=1000):
    # 对数据进行FFT变换
    fft_result = np.fft.fft(data, axis=-1)  # 对最后一个维度进行FFT
    frequencies = np.fft.fftfreq(data.shape[-1], 1/sampling_frequency)

    # 计算频谱的幅度
    magnitude = np.abs(fft_result)

    # 找到直流分量的位置
    dc_index = np.where(frequencies == 0)[0][0]

    # 去除直流分量
    magnitude_without_dc = magnitude.copy()
    magnitude_without_dc[:, :, :, dc_index] = 0

    # 对去除直流分量后的频谱进行反FFT，获得时域信号
    signal_without_dc = np.fft.ifft(magnitude_without_dc, axis=-1)
    
    return signal_without_dc


#csi数据插值，一般来说csi数据的丢包率在5%-10%之间，此处要根据数据的时间戳信息进行修补
#首先要把要插值的数据传进来，然后把横坐标时间戳传进来（此处也是从read_bf_file读出来的信息）
#最大时间间隔和最小时间间隔对于每个数据集都不同，需要手动定义
#调试可以知道：当我们设定发包参数为1000的时候(也就是一秒钟发1000个包，两个包之间相隔10^-3秒)，会在1000上下跳动（波动比较大）
def interpolate_and_remove_csi(csi_amp, timestamps,max_time,low_time):
    # 初始化插值器,此处采用线性插值
    interpolator = interp1d(timestamps, csi_amp, kind='linear', axis=0)

    # 初始化结果列表
    new_csi_amp = [csi_amp[0]]
    new_timestamps = [timestamps[0]]
    
    #建立时间戳差距表(用于调试)
    timestamps_diff_too_high = []
    timestamps_diff_too_low = []
    
    # 遍历时间戳和 CSI 数据
    for i in range(1, len(timestamps)):
        time_diff = timestamps[i] - timestamps[i - 1]
        #print(time_diff)

        if time_diff > max_time:
            timestamps_diff_too_high.append(time_diff)
            
            # 时间差异大于最大阈值，进行插值
            num_interp_points = int(time_diff / max_time)
            interp_timestamps = np.linspace(timestamps[i - 1], timestamps[i], num_interp_points + 1)
            interp_csi = interpolator(interp_timestamps)
            
            # 将插值后的数据添加到结果中
            new_csi_amp.extend(interp_csi[1:])  # 跳过第一个数据，避免重复
            new_timestamps.extend(interp_timestamps[1:])  # 跳过第一个时间戳

        elif time_diff < low_time:
            timestamps_diff_too_low.append(time_diff)
            # 时间差异小于最小阈值，进行去重
            continue
        else:
            # 时间差异在最小和最大阈值之间，保留原始数据
            new_csi_amp.append(csi_amp[i])
            new_timestamps.append(timestamps[i])
            
    return np.array(new_csi_amp), np.array(new_timestamps),timestamps_diff_too_high,timestamps_diff_too_low


#使用hampel滤波器对离群值进行去除，window_size为滑动窗口的大小，n_sigma为中位数绝对偏差的倍数，用来确定阈值
#基本原理：在一个滑动窗口内计算其所有元素的中位数值，并用中位数绝对值估计各个元素和其的标准差，如果超过sigma个标准差，那就用中位数值替换其
def hampel_filter(csi_data, kernel_size=3, threshold_multiplier=3.0):
    filtered_csi_data = np.zeros_like(csi_data)

    for i in range(csi_data.shape[1]):
        for j in range(csi_data.shape[2]):
            for k in range(csi_data.shape[3]):
                subcarrier_data = csi_data[:, i, j, k]
                median = medfilt(subcarrier_data, kernel_size=kernel_size)
                median_abs_deviation = np.median(np.abs(subcarrier_data - median))
                threshold = threshold_multiplier * median_abs_deviation
                outliers = np.abs(subcarrier_data - median) > threshold
                filtered_csi_data[:, i, j, k] = np.where(outliers, median, subcarrier_data)

    return filtered_csi_data



