import numpy as np
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

'''
PCA降维
输入为原始数据(二维矩阵)和希望保留的维度百分比
返回值为降维后的数据和被选择的特征向量
perform_pca实现使用pca算法进行数据降维，其余两个函数绘制选择后的主成分子载波图和主成分方差贡献表
'''
def perform_pca(csi_data, explained_variance_ratio_threshold=0.95):
    # # 去中心化
    # mean_vals = np.mean(csi_data, axis=0)
    # centered_data = csi_data - mean_vals
    
    # 计算协方差矩阵
    cov_matrix = np.cov(csi_data, rowvar=False)
    
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 从大到小排序特征值
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # 计算特征值的比例
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    
    #plot_explained_variance_ratio(explained_variance_ratio)
    
    # 选择主成分数量
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    num_components = np.argmax(cumulative_variance_ratio >= explained_variance_ratio_threshold) + 1
    
    # 选择前num_components个特征向量
    selected_eigenvectors = eigenvectors[:, :num_components]
    
    # 计算降维结果
    reduced_data = np.dot(csi_data, selected_eigenvectors)
    
    return reduced_data, selected_eigenvectors, explained_variance_ratio[:num_components]


#绘制不同成分的累计方差贡献表，输入值为训练好的pca模型
def plot_explained_variance_ratio(explained_variance_ratio):
    # 计算累积方差贡献比
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # 绘制方差贡献比图
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center',
            label='Individual explained variance ratio')
    plt.step(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, where='mid',
             label='Cumulative explained variance ratio', color='red')
    
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio of Principal Components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
def plot_data(timestamps, reduced_data, selected_eigenvectors, top_k=5):
    # 只显示前top_k个主成分
    for i in range(min(top_k, reduced_data.shape[1])):
        # 找到该特征向量中的最大元素对应的索引
        most_important_index = np.argmax(np.abs(selected_eigenvectors[:, i]))
        
        # 确定对应的发送天线、接收天线和子载波
        #np.unraval_index获取一个索引值在多维数组中的位置
        tx_antenna, rx_antenna, subcarrier = np.unravel_index(most_important_index, (3, 3, 30))
        
        # 为该主成分的线添加标签
        label = f"Component {i + 1} (TX: {tx_antenna + 1}, RX: {rx_antenna + 1}, Subcarrier: {subcarrier + 1})"
        plt.plot(timestamps, reduced_data[:, i], label=label)

    plt.xlabel("Timestamps")
    plt.ylabel("CSI Amplitude")
    plt.legend()
    plt.show()
    
    
'''
DTW算法实现子载波选择
(计算量较大)先别用，后面再优化一下
'''
#快速计算DTW距离
def fastdtw_distance(matrix1, matrix2):
    distance, _ = fastdtw(matrix1, matrix2, dist=euclidean)
    return distance

#计算两个矩阵的DTW距离
def dtw_distance(matrix1, matrix2):
    return dtw(matrix1, matrix2)

#按行求和，计算每个天线和其他天线的DTW距离
def compute_dtw_matrix(csi_data):
    num_subcarriers = csi_data.shape[3]
    dtw_matrix = np.zeros((num_subcarriers, num_subcarriers))
    
    csi_data_reduced = csi_data[:100]

    for i in range(num_subcarriers):
        for j in range(num_subcarriers):
            matrix1 = csi_data_reduced[..., i].flatten()
            matrix2 = csi_data_reduced[..., j].flatten()
            
            matrix1 = matrix1.reshape((matrix1.shape[0],1))
            matrix2 = matrix2.reshape((matrix2.shape[0],1))
            #print(matrix1.shape)
            #print(matrix2.shape)
            
            dtw_matrix[i, j] = fastdtw_distance(matrix1, matrix2)

    return dtw_matrix

#按照给定的DTW距离矩阵完成子载波选择，返回值为被选择的子载波索引号
def select_subcarriers(dtw_matrix, num_to_select=10):
    selection_indices = np.argsort(np.sum(dtw_matrix, axis=1))
    return selection_indices[:num_to_select]


# 绘制原始子载波和选出的子载波的对比图
def plot_subcarriers_comparison(csi_data, selected_subcarriers):
    num_subcarriers = csi_data.shape[3]
    all_subcarriers = np.arange(num_subcarriers)

    plt.figure(figsize=(10, 6))
    plt.bar(all_subcarriers, np.mean(csi_data.reshape(-1, num_subcarriers), axis=0), label='All Subcarriers')
    plt.bar(selected_subcarriers, np.mean(csi_data.reshape(-1, num_subcarriers)[:, selected_subcarriers], axis=0), label='Selected Subcarriers', color='red')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Average Amplitude')
    plt.title('Comparison of All Subcarriers vs Selected Subcarriers')
    plt.legend()
    plt.show()





