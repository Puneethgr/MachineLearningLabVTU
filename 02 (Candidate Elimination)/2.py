# Program 2 (Candidate Elimination):

import pandas as pd

data = pd.read_csv("EnjoySport.csv", header = None)

n = len(data.columns) - 1
s = ['0' for _ in range(n)]
g = [['?' for _ in range(n)]]

print("Initial Values: ")
print("Specific Hypothesis: ", s)
print("General Hypothesis: ", g)

def consistencyOfG(row, generalHypothesisSet):
    newG = []
    for h in generalHypothesisSet:
        consistent = True
        for colIndex in range(len(h)):
            if h[colIndex] == '?':
                continue
            elif h[colIndex] != row[colIndex]:
                consistent = False
                break
        if consistent:
            newG.append(h)
    return newG


for index, row in data.iterrows():
    print("------------")
    print("Row {} : {}".format(index+1, row.values))
    if row[len(row)-1] == "Yes":
        g = consistencyOfG(row,g)
        for colIndex in range(len(row)-1):
            if s[colIndex] == '0':
                s[colIndex] = row[colIndex]
            elif s[colIndex] != row[colIndex]:
                s[colIndex] = '?'
    else:
        newG = []
        moreGeneral = []
        for i in range(n):
            for h in g:
                if h[i] == '?' and s[i] != row[i]:
                    newH = list(h)
                    newH[i] = s[i]
                    newG.append(newH)
                    moreGeneral.append(h)

        for generalH in moreGeneral:
            if generalH in newG:
                newG.remove(generalH)

        g = newG
        
    print("Specific Hypothesis: ", s)
    print("General Hypothesis: ", g)