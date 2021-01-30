# Program 8 (KMeans and Expectation–maximization [EM])

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans 
from sklearn.mixture import GaussianMixture

iris = datasets.load_iris()
X_test, X_train, y_test, y_train = train_test_split(iris.data, iris.target, 
                                                    train_size = 0.8, test_size = 0.2, 
                                                    random_state = 100)

model = KMeans(n_clusters = 3)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print("KMeans Accuracy : ", accuracy_score(y_test, predictions))

model2 = GaussianMixture(n_components = 3)
model2.fit(X_train,y_train)
predictions2 = model2.predict(X_test)
print("EM Accuracy : ", accuracy_score(y_test, predictions2))