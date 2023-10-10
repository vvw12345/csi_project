from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import pandas as pd


'''
Linear Discriminant Analysis线性判别分析实现学习分类
'''
def lda_classification(readpath):    
    # 从CSV文件中读取数据
    data = pd.read_csv(readpath)

    # 提取特征和标签
    features = data.drop(["labels"], axis=1)  # 假设标签在最后一列，此处使用drop函数去除标签列
    labels = data["labels"]

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # 创建一个LDA分类器
    lda = LinearDiscriminantAnalysis(
        solver="svd",
        shrinkage=None,
        priors=None,
        n_components=None,
        store_covariance=False,
        tol=0.0001,
    )

    # 训练分类器
    lda.fit(X_train, y_train)

    # 使用分类器对测试集进行预测
    y_pred = lda.predict(X_test)

    # 计算分类器的准确率
    accuracy = accuracy_score(y_test, y_pred)

    # 输出准确率
    print("Accuracy:", accuracy)




