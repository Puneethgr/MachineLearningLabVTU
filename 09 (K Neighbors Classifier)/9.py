# Program 9 (K Neighbors Classifier)

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target,
                                                    train_size=0.8,test_size=0.2,
                                                    random_state=100)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
predictions = model.predict(X_test) 

print("Confusion Matrix : ")
print(confusion_matrix(y_test, predictions))
print("Classification Report : ")
print(classification_report(y_test, predictions))