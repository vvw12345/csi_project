import os
import pandas as pd

"""
    将特征和标签存储到csv文件中。
"""
def save_features_to_csv(features,csv_file):
    # 创建DataFrame
    df = pd.DataFrame(features).T
    #df['labels'] = label
    
    # 判断文件是否存在，如果存在则追加，否则创建新文件
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False,index=False)
    else:
        df.to_csv(csv_file, mode='w',index=False)  #不存入索引列 保留特征索引

   
#从文件名字中提取出关键字打上标签
'''
数据命名格式：xxx_xxx_xx
第几次实验_什么动作_第几次测试
这里有个很大的问题，在提取标签的时候要注意只把单独的文件名传入做分割，而不是完整的文件寻址路径
因为文件路径可能也会有下划线，从而导致分割出现问题，最终导致没有标签
'''    
def extract_label_from_filename(filename):
    parts = filename.split("_")
    #print(parts)
    if len(parts) == 3:  # 防止分割异常的情况出现
        return parts[1]
    else:
        return None