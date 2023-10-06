import os
import pandas as pd

"""
    将特征和标签存储到csv文件中。
"""
def save_features_to_csv(features, label, csv_file):
    # 创建DataFrame
    df = pd.DataFrame(features).T
    df['labels'] = label
    
    # 判断文件是否存在，如果存在则追加，否则创建新文件
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False)
    else:
        df.to_csv(csv_file, mode='w')

   
#从文件名字中提取出关键字打上标签
'''
数据命名格式：xxx_xxx_xx
第几次实验_什么动作_第几次测试
'''     
def extract_label_from_filename(filename):
    parts = filename.split("_")
    if len(parts) == 3:  # 防止分割异常的情况出现
        return parts[1]
    else:
        return None