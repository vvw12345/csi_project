import matplotlib.pyplot as plt
import os
from data_loading.csi_getting import *
from data_loading.data_reading import *
from data_loading.csi_csv import *
from data_processing.preprocessing import *
from data_processing.denoising import *
from data_processing.dimension_reduction import *
from data_processing.movement_detective import *

##注：本函数将为整个csi数据的流程：数据读入->预处理->特征提取->机器学习

path = r"./data_loading/mytest.dat"
directory = r"./datas"
output_directory = r"./output"
#path2 = r"./data_loading/output.csv"

# #如果输出目录不存在就创建一个
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)
    
# file_list = os.listdir(directory)

bf = read_bf_file(path)

csi_list = list(map(get_scale_csi,bf))
timestamps = [entry['timestamp_low'] for entry in bf]
#print(timestamps)
#save_csv(csi_list,path2)
csi_np = (np.array(csi_list))

#计算幅值
csi_amp = np.abs(csi_np)
print(csi_amp.shape)
#数据去中心化
centerize_csi_amp = centerize_csi(csi_amp)

#数据插值和去重获得新的赋值数据（此处我们选择数据前后的10%作为插值和去重的临界点）
interpolated_csi_amp,new_timestamps,timestamps_too_high,timestamps_too_low = interpolate_and_remove_csi(centerize_csi_amp,timestamps,1100,900)
# print(timestamps_too_high)
# print(len(timestamps_too_high))
# print(len(timestamps_too_low))

#hampel滤波器去除离群值
filtered_csi_amp = hampel_filter(interpolated_csi_amp,3,3)

#设置巴特沃斯滤波器相关参数
#截止频率：一种常用的经验法则是选择截止频率的值，使其位于采样频率的一半以下，通常在0.2倍采样频率到0.4倍采样频率之间。
cutoff_frequency = 400.0  # 截止频率
sampling_frequency = 1000.0  # 采样频率

butterworth_denoised_csi_amp = denoise_csi_data(filtered_csi_amp, cutoff_frequency, sampling_frequency)
#print(butterworth_denoised_csi_amp.shape)
#使用小波去噪
wavelet_denoised_csi_amp = apply_wavelet_denoising(filtered_csi_amp, wavelet='db4', level=3)
#使用savitzky_golay滤波器
savitzky_golay_denoised_csi_amp = apply_savitzky_golay_filter(filtered_csi_amp, window_length=5, polynomial_order=2)

'''
基于滑动方差的动作提取
'''
# 使用函数
variances = moving_variance(filtered_csi_amp, 5)
print(max(variances))
print(min(variances))
activities = detect_activity(variances)
#print(activities)
plot_csi_with_activity(filtered_csi_amp, 5, activities)


'''基于DTW算法的子载波选择
#获得DTW矩阵
#dtw_mat = compute_dtw_matrix(savitzky_golay_denoised_csi_amp)
#dtw_select = select_subcarriers(dtw_mat)
#print(dtw_select)
'''

'''基于PCA算法的数据降维
#采用pca数据降维，在数据降维之前首先要把数据压缩为2维数组
reshaped_data = butterworth_denoised_csi_amp.reshape(savitzky_golay_denoised_csi_amp.shape[0], -1)
pca_reduced_csi_amp,selected_eigenvectors, explained_variance_ratio = perform_pca(reshaped_data,0.95)
#print(pca_reduced_csi_amp.shape)
# 绘制图形，并将天线和子载波信息添加到图上
plot_data(range(len(pca_reduced_csi_amp)),pca_reduced_csi_amp,selected_eigenvectors,7)
'''



# most_important_component = np.argmax(explained_variance_ratio)
# most_important_eigenvector = selected_eigenvectors[:, most_important_component]
# most_important_index = np.argmax(np.abs(most_important_eigenvector))
    
# tx_antenna, rx_antenna, subcarrier = np.unravel_index(most_important_index, (3, 3, 30))



'''
# 获取预训练好的PCA模型的特征值
eigenvalues = pca_model.explained_variance_
# 获取预训练好的PCA模型的方差贡献率
explained_variance_ratio = pca_model.explained_variance_ratio_
data_list = list(zip(eigenvalues, explained_variance_ratio))
# 打印特征值和方差贡献率列表
print("特征值和方差贡献率列表:")
print(data_list)
'''
#plot_variance_explained(pca_model)



# # 创建一个新的图形
#plt.figure(figsize=(10, 5))

# 绘制原始 CSI 幅度数据
#plt.plot(range(len(csi_amp)), csi_amp[:, 0, 0, 5], label='init__csi', color='b')
# #绘制去中心化之后的csi数据
# #plt.plot(range(len(centerize_csi_amp)), centerize_csi_amp[:, 0, 0, 6], label='centerize__csi', color='g')
# #绘制数据插值之后的csi数据
# #plt.plot(range(len(interpolated_csi_amp)), interpolated_csi_amp[:, 0, 0, 6], label='interpolated_csi', color='r')
# #绘制汉普滤波器之后得到的csi数据
#plt.plot(range(len(filtered_csi_amp)), filtered_csi_amp[:, 0, 0, 5], label='filtered_csi', color='g')
# #plt.plot(range(len(signal_without_dc)),np.real(signal_without_dc[:,0,0,6]),color = 'g')
# plt.plot(range(len(savitzky_golay_denoised_csi_amp)), savitzky_golay_denoised_csi_amp[:, 0, 0, 6], label='savitzky_golay', color='r')
# plt.plot(range(len(wavelet_denoised_csi_amp)), wavelet_denoised_csi_amp[:, 0, 0, 6], label='wavelet_csi', color='m')
# plt.plot(range(len(butterworth_denoised_csi_amp)), butterworth_denoised_csi_amp[:, 0, 0, 6], label='butterworth_csi', color='y')
# plt.xlabel('timestamps')
# plt.ylabel('CSI')
# plt.title('CSI Data Comparison')
# plt.legend()
#plt.show()







