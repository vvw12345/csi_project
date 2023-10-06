import numpy as np
from scipy.stats import iqr, entropy
from collections import OrderedDict

'''
特征值计算
标准差：CSI幅度方差的算数平方根
活动时长：CSI滑动方差超过阈值时间（此处应该指动作提取的部分）
中位数绝对偏差：CSI幅度绝对偏差的中位数
四分位距：CSI幅度的四等分中第三个四分位上的值和第一个四分位上的值的差
信息熵：CSI幅度信息含量的量化指标
最大值，最小值，差值
'''
import numpy as np
from scipy.stats import entropy

def calculate_features(csi_data,label):
    num_timestamps, num_ntx, num_nrx, num_subcarriers = csi_data.shape
    all_features = OrderedDict()

    # 遍历每个发射天线和接收天线
    for t in range(num_ntx):
        for r in range(num_nrx):
            
            # 遍历每个子载波
            for i in range(num_subcarriers):
                subcarrier_data = csi_data[:, t, r, i].reshape(-1)  

                # 计算特征值
                std_deviation = np.std(subcarrier_data)  # 标准差
                activity_duration = np.count_nonzero(subcarrier_data)  # 活动时长（非零元素的个数）
                mad = np.median(np.abs(subcarrier_data - np.median(subcarrier_data)))  # 中位数绝对偏差
                quartile_25, quartile_75 = np.percentile(subcarrier_data, [25, 75])  # 四分位距
                iqr_value = quartile_75 - quartile_25  # 四分位距
                
                # 获取非零元素
                non_zero_data = subcarrier_data[subcarrier_data > 0]
                # 计算非零元素的概率
                prob = non_zero_data / non_zero_data.sum()
                info_entropy = -np.sum(prob * np.log(prob))# 信息熵
                
                max_value = np.max(subcarrier_data)  # 最大值
                min_value = np.min(subcarrier_data)  # 最小值
                difference = max_value - min_value  # 差值

                # 存储特征值到字典中
                key = f'NTX_{t+1}_NRX_{r+1}_Subcarrier_{i+1}'
                features = {
                    '标准差': std_deviation,
                    '活动时长': activity_duration,
                    '中位数绝对偏差': mad,
                    '四分位距': iqr_value,
                    '信息熵': info_entropy,
                    '最大值': max_value,
                    '最小值': min_value,
                    '差值': difference,
                    'labels': label
                }

    all_features[key] = features
    return all_features

