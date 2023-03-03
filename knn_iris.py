from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris.data[:3]

iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df['target'] = pd.Series(iris.target)
iris_df.head(10)

x = iris_df.iloc[:, :4]
y = iris_df.iloc[:, -1]

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def iris_knn(x, y, k) :
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    return metrics.accuracy_score(y_test, y_pred)

k = 3
scores = iris_knn(x, y, k)
print("n_neighbors가 {0:d}일 때 정확도 : {1:.3f}".format(k, scores))