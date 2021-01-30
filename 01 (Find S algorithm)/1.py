# Program 1 (Find S):
 
import pandas as pd 
 
data = pd.read_csv("EnjoySport.csv", header = None)
print(data)
 
numberAttributes = len(data.columns)-1
hypothesis = ['0' for _ in range(numberAttributes)]
 
print("hypothesis 0 : ", hypothesis)
for index, row in data.iterrows():
    if row[len(row)-1] == "Yes":
        for colIndex in range(len(row)-1):
            if hypothesis[colIndex] == '0':
                hypothesis[colIndex] = row[colIndex]
            elif hypothesis[colIndex] != row[colIndex]:
                hypothesis[colIndex] = '?'
        print("hypothesis {} : {}".format(index+1, hypothesis))
 
print("Final hypothesis: ", hypothesis)