# Program 5 (Naive Bayes on Golf dataset)

from pprint import pprint
import pandas as pd
from collections import Counter

df = pd.read_csv("golf.csv")
targetAttribute = "label"

attributes = list(df.columns)
attributes.remove(targetAttribute)

train = df.sample(frac=0.6,random_state=100)
test = df.drop(train.index)

table = dict()
priorProb = dict()

for attr_val, data_subset in train.groupby(targetAttribute):
    valueCount = dict()
    count = 0
    for attribute in attributes:
        counter = Counter(x for x in data_subset[attribute])
        valueCount[attribute] = dict(counter)
        count = sum(counter.values())
    table[attr_val] = valueCount
    priorProb[attr_val] = count

print("------------------------------")
print("The Resultant table is :")
pprint(table)

totalSize = test[targetAttribute].count()
correctPredictions = 0

for k, row in test.iterrows():

    rowTuple = dict(row)
    keyList = [x for x in rowTuple.keys() if x != targetAttribute]
    labelList = list()
    posterioriList = list()

    for label in table.keys():
        posteriori = 1.0
        for key in keyList:
            y = table[label][key]
            x = rowTuple.get(key)
            if x in y.keys():
                posteriori *= (y[x]/sum(y.values()))
        posteriori *= priorProb[label]
        
        labelList.append(label)
        posterioriList.append(posteriori)

    maxProbInd = posterioriList.index(max(posterioriList))
    if rowTuple[targetAttribute] == labelList[maxProbInd]:
        correctPredictions += 1

print("------------------------------")
print("Number of Correct Predictions : ",correctPredictions)
print("Number of Samples: ", totalSize)
print("Accuracy:",100.0*correctPredictions/totalSize)