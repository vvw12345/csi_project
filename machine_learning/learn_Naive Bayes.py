import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def train_naive_bayes_classifier(filepath):
    """
    使用高斯朴素贝叶斯分类器训练模型并计算准确率

    参数:
    filepath (str): CSV文件路径

    返回:
    float: 分类器的准确率
    """
    # 读取数据
    data = pd.read_csv(filepath)

    # 提取特征和标签
    features = data.drop(["labels"], axis=1)  # 假设标签在最后一列
    labels = data["labels"]

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # 创建高斯朴素贝叶斯分类器
    gnb = GaussianNB()

    # 训练分类器
    gnb.fit(X_train, y_train)

    # 使用分类器对测试集进行预测
    y_pred = gnb.predict(X_test)

    # 计算分类器的准确率
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# 调用函数
readpath = './feature_output/output.csv'
accuracy = train_naive_bayes_classifier(readpath)
print("Accuracy:", accuracy)
