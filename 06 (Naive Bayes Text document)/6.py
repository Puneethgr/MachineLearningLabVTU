# Program 6 (Naive Bayes on Text document)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("imdb_labelled.txt", sep = "\t", index_col = None)
print(df)
X_train, X_test, y_train, y_test = train_test_split(df['Text'],df['Label'], 
                                                    train_size = 0.8, test_size = 0.2, 
                                                    random_state = 100)

cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

model = MultinomialNB()
model.fit(X_train_cv, y_train)
predictions = model.predict(X_test_cv)

print("Accuracy score: ", accuracy_score(y_test, predictions))
print("Precision score: ", precision_score(y_test, predictions))
print("Recall score: ", recall_score(y_test, predictions))