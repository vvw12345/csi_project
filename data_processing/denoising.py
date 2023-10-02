import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal import savgol_filter
import pywt

'''
巴特沃斯滤波器去噪部分
'''
#设计Butterworth低通滤波器
#cutoff是截止频率，fs是采样频率，order滤波器阶数
def butter_lowpass(cutoff, fs, order=8):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a #分子系数和分母系数

#应用Butterworth低通滤波器进行滤波
def butter_lowpass_filter(data, cutoff, fs, order=8):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

#cutoff_frequency是截止频率，sampling_frequency是采样频率
def denoise_csi_data(csi_data, cutoff_frequency, sampling_frequency):
    denoised_csi_data = np.zeros_like(csi_data)
    
    for i in range(csi_data.shape[1]):
        for j in range(csi_data.shape[2]):
            for k in range(csi_data.shape[3]):
                subcarrier_data = csi_data[:, i, j, k]
                denoised_subcarrier = butter_lowpass_filter(subcarrier_data, cutoff_frequency, sampling_frequency)
                denoised_csi_data[:, i, j, k] = denoised_subcarrier
    
    return denoised_csi_data


'''
小波去噪部分
'''
# 定义小波去噪函数
def denoise_wavelet(subcarrier_data, wavelet='db4', level=1):
    coeffs = pywt.wavedec(subcarrier_data, wavelet, level=level)
    threshold = np.std(coeffs[-level]) * np.sqrt(2 * np.log(len(subcarrier_data)))
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    denoised_subcarrier = pywt.waverec(coeffs_thresholded, wavelet)
    return denoised_subcarrier

# 定义小波去噪的封装函数
def apply_wavelet_denoising(csi_data, wavelet='db4', level=1):
    denoised_csi_data = np.zeros_like(csi_data)
    
    for i in range(csi_data.shape[1]):
        for j in range(csi_data.shape[2]):
            for k in range(csi_data.shape[3]):
                subcarrier_data = csi_data[:, i, j, k]
                denoised_subcarrier = denoise_wavelet(subcarrier_data, wavelet=wavelet, level=level)
                denoised_csi_data[:, i, j, k] = denoised_subcarrier
    
    return denoised_csi_data


'''
Savitzky-Golay滤波部分
'''
# 定义Savitzky-Golay滤波函数
def savitzky_golay_filter(subcarrier_data, window_length=5, polynomial_order=2):
    smoothed_subcarrier = savgol_filter(subcarrier_data, window_length, polynomial_order)
    return smoothed_subcarrier

# 定义Savitzky-Golay滤波的封装函数
def apply_savitzky_golay_filter(csi_data, window_length=5, polynomial_order=2):
    smoothed_csi_data = np.zeros_like(csi_data)
    
    for i in range(csi_data.shape[1]):
        for j in range(csi_data.shape[2]):
            for k in range(csi_data.shape[3]):
                subcarrier_data = csi_data[:, i, j, k]
                smoothed_subcarrier = savitzky_golay_filter(subcarrier_data, window_length, polynomial_order)
                smoothed_csi_data[:, i, j, k] = smoothed_subcarrier
    
    return smoothed_csi_data


