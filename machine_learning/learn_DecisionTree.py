import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

readfile = "./feature_output/output.csv"

def decision_tree_classification(readpath):
    # 1. 读取csv文件
    data = pd.read_csv(readpath)

    # 2. 分割数据和标签
    X = data.drop(["labels"], axis=1)
    y = data['labels']

    # 3. 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 4. 使用决策树算法
    clf = DecisionTreeClassifier()  # 可以调整DecisionTreeClassifier内的参数以优化模型
    clf.fit(X_train, y_train)

    # 5. 预测并评估
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    print("训练集准确率: {:.2f}%".format(accuracy_score(y_train, y_train_pred) * 100))
    print("测试集准确率: {:.2f}%".format(accuracy_score(y_test, y_test_pred) * 100))
    #print(confusion_matrix(y_test, y_test_pred))
    #print(classification_report(y_test, y_test_pred))







