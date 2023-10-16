import matplotlib.pyplot as plt
import os
import glob
from data_loading.csi_getting import *
from data_loading.data_reading import *
from data_loading.csi_csv import *
from data_processing.preprocessing import *
from data_processing.denoising import *
from data_processing.dimension_reduction import *
from data_processing.movement_detective import *
from feature_collecting.feature_collecting import *

##注：本函数将为整个csi数据的流程：数据读入->预处理->特征提取->机器学习

# 获取当前文件的路径
current_path = os.path.dirname(os.path.abspath(__file__))
#print(current_path)  #此处成功获取到我的路径 d:\项目\github\csi_project

# dataset目录的路径
dataset_path = os.path.join(current_path, 'dataset')
output_directory = os.path.join(current_path,'feature_output')
output_path = os.path.join(output_directory,'output_init.csv')

#如果输出目录不存在就创建一个
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

#全流程处理函数
def processing_function(file):
    filename = os.path.basename(file)
    '''
    数据读入部分
    '''
    bf = read_bf_file(file)
    csi_list = list(map(get_scale_csi,bf))
    timestamps = [entry['timestamp_low'] for entry in bf]
    csi_np = (np.array(csi_list))
    
    #计算幅值和相位
    csi_amp = np.abs(csi_np)
    print(csi_amp.shape)
    csi_phase = np.unwrap(np.angle(csi_np),axis = 1)
    
    
    #数据去中心化
    centerize_csi_amp = centerize_csi(csi_amp)
    
    # #数据插值和去重获得新的赋值数据（此处我们选择数据前后的10%作为插值和去重的临界点）
    (interpolated_csi_amp,
      new_timestamps,timestamps_too_high,
     timestamps_too_low) = interpolate_and_remove_csi(centerize_csi_amp,timestamps,1100,900)
    
    # #hampel滤波器去除离群值
    filtered_csi_amp = hampel_filter(interpolated_csi_amp,3,3)
    
    # '''
    # 巴特沃斯滤波器，小波去噪，savitzky_golay滤波器降噪部分
    # '''
    # #设置巴特沃斯滤波器相关参数
    # #截止频率：一种常用的经验法则是选择截止频率的值，使其位于采样频率的一半以下，通常在0.2倍采样频率到0.4倍采样频率之间。
    # cutoff_frequency = 400.0  # 截止频率
    # sampling_frequency = 1000.0  # 采样频率

    # butterworth_denoised_csi_amp = denoise_csi_data(filtered_csi_amp, cutoff_frequency, sampling_frequency)
    # #print(butterworth_denoised_csi_amp.shape)

    # #使用小波去噪
    # #wavelet_denoised_csi_amp = apply_wavelet_denoising(filtered_csi_amp, wavelet='db4', level=3)
    
    # #使用savitzky_golay滤波器
    # #savitzky_golay_denoised_csi_amp = apply_savitzky_golay_filter(filtered_csi_amp, window_length=5, polynomial_order=2)

    # '''
    # 基于滑动方差的动作提取
    # 目前动作识别效果还是一般般，等待优化
    # '''
    # # # 使用函数
    # # target_subcarrier_idx = 6
    # # variances = moving_variance(filtered_csi_amp, target_subcarrier_idx)
    # # activities = detect_activity(variances)
    # # intervals = get_activity_intervals(activities)
    
    '''
    特征值提取部分
    '''
    label = extract_label_from_filename(filename)
    features = calculate_features(csi_amp,label)
    #print(features)
    
    # 存储到csv
    save_features_to_csv(features,output_path)
    
    '''
    行为识别——机器学习模型拟合部分
    '''
    pass


    
# 遍历lab01_data和lab02_data文件夹
for lab_folder in ['lab01_data', 'lab02_data']:
    # 构造完整的文件夹路径
    lab_path = os.path.join(dataset_path, lab_folder)
    # 查找文件夹中的所有文件
    files = glob.glob(os.path.join(lab_path, '*'))
    for file in files:
        # 对每个文件调用处理函数
        processing_function(file)
        






























