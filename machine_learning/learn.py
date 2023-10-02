import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 从CSV文件中读取数据
data = pd.read_csv('result_lab01.csv')

# 提取特征和标签
features = data.drop(['label'], axis=1)  # 假设标签在最后一列，此处使用drop函数去除标签列
labels = data['label']

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42)

# 创建一个支持向量机分类器
svm = SVC(kernel='linear')

# 训练分类器
svm.fit(X_train, y_train)

# 使用分类器对测试集进行预测
y_pred = svm.predict(X_test)

# 计算分类器的准确率
accuracy = accuracy_score(y_test, y_pred)

# 输出准确率
print("Accuracy:", accuracy)
