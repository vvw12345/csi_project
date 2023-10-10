import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

'''
KNN近邻算法实现分类 
读入文件路径 以及KNN中的k值
问题：处理前比处理后准确率高 训练集在90%左右 测试集在80%左右
'''
def knn_classification(readpath, k=3):
    # 1. 读取csv文件
    data = pd.read_csv(readpath)

    # 2. 分割数据和标签
    X = data.drop(["labels"], axis=1)
    y = data['labels']

    # 3. 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 4. 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5. 使用KNN算法
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(X_train, y_train)

    # 6. 预测并评估
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    print("训练集准确率: {:.2f}%".format(accuracy_score(y_train, y_train_pred) * 100))
    print("测试集准确率: {:.2f}%".format(accuracy_score(y_test, y_test_pred) * 100))
    
    
readpath = "./feature_output/output_init.csv"
# 1. 读取csv文件
data = pd.read_csv(readpath)

# 2. 分割数据和标签
X = data.drop(["labels"], axis=1)
y = data['labels']

# 3. 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. 使用KNN算法
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, y_train)

# 6. 预测并评估
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

print("训练集准确率: {:.2f}%".format(accuracy_score(y_train, y_train_pred) * 100))
print("测试集准确率: {:.2f}%".format(accuracy_score(y_test, y_test_pred) * 100))


