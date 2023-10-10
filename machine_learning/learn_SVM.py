import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

'''
SVM支持向量机实现分类
目前主要问题：1.对原始数据训练时间太长,线性核训练不出来，使用高斯核的时候加大惩罚系数可以达到73%
             2.对于预处理数据，线性核的准确率是74%；而高斯核表现不佳，即使把惩罚系数加大，最高也只能到70%左右
数据预处理过程中使用了行向量标准化，出现了0均值，导致高斯核表现不佳
'''
def svm_classification(readpath):
    # 从CSV文件中读取数据
    data = pd.read_csv(readpath)

    # 输出数据列信息
    print(data.columns)

    # 提取特征和标签
    features = data.drop(['labels'], axis=1)  # 假设标签在最后一列，此处使用drop函数去除标签列
    labels = data['labels']

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


readpath = './feature_output/output_init.csv'
# 从CSV文件中读取数据
data = pd.read_csv(readpath)

# 输出数据列信息
print(data.columns)

# 提取特征和标签
features = data.drop(['labels'], axis=1)  # 假设标签在最后一列，此处使用drop函数去除标签列
labels = data['labels']

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42)

# 创建一个支持向量机分类器
svm = SVC(C = 1000,kernel='rbf')

# 训练分类器
svm.fit(X_train, y_train)

# 使用分类器对测试集进行预测
y_pred = svm.predict(X_test)

# 计算分类器的准确率
accuracy = accuracy_score(y_test, y_pred)

# 输出准确率
print("Accuracy:", accuracy)