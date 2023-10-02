import numpy as np
from scipy.stats import iqr, entropy

'''
特征值计算
标准差：CSI幅度方差的算术平方根
活动时长：CSI滑动方差超过阈值时间

'''
def calculate_features(csi_data):
    features = {}
    num_timestamps, num_subcarriers = csi_data.shape
    
    # 遍历每个子载波
    for i in range(num_subcarriers):
        # 将子载波数据拉平成一维数组
        subcarrier_data = csi_data[:, i].reshape(-1)  
        
        # 计算特征值
        std_deviation = np.std(subcarrier_data)  # 标准差
        activity_duration = np.count_nonzero(subcarrier_data)  # 活动时长（非零元素的个数）
        mad = np.median(np.abs(subcarrier_data - np.median(subcarrier_data)))  # 中位数绝对偏差
        quartile_25, quartile_75 = np.percentile(subcarrier_data, [25, 75])  # 四分位距
        iqr_value = quartile_75 - quartile_25  # 四分位距
        info_entropy = entropy(subcarrier_data)  # 信息熵
        max_value = np.max(subcarrier_data)  # 最大值
        min_value = np.min(subcarrier_data)  # 最小值
        difference = max_value - min_value  # 差值
        
        # 存储特征值到字典中
        features[f'Subcarrier_{i+1}'] = {
            '标准差': std_deviation,
            '活动时长': activity_duration,
            '中位数绝对偏差': mad,
            '四分位距': iqr_value,
            '信息熵': info_entropy,
            '最大值': max_value,
            '最小值': min_value,
            '差值': difference
        }
    
    return features

