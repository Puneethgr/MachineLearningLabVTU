# Program 3 (ID3 algorithm)

import pandas as pd
import math
from collections import Counter
from pprint import pprint

df = pd.read_csv("tennis.csv")
targetAttribute = 'PlayTennis'

def entropy(probs):
    return sum( [-prob*math.log(prob, 2) for prob in probs] )

def entropy_of_list(targetValues):
    counter = Counter(targetValues)
    numberInstances = len(targetValues)*1.0     
    probs = [x / numberInstances for x in counter.values()]
    return entropy(probs)

def information_gain(df, splitAttribute, targetAttribute, trace=0): 
    df_split = df.groupby(splitAttribute)
    nobs = len(df.index) * 1.0
    df_agg_ent = df_split.agg({targetAttribute : [entropy_of_list, lambda x: len(x)/nobs] })[targetAttribute] 
    df_agg_ent.columns = ['Entropy', 'PropObservations'] 
    new_entropy = sum(df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
    old_entropy = entropy_of_list(df[targetAttribute]) 
    return old_entropy - new_entropy

def id3(df, targetAttribute, attributeNames, default_class=None):
    counter = Counter(x for x in df[targetAttribute])
    if len(counter) == 1:         
        return next(iter(counter)) 
    elif df.empty or (not attributeNames): 
             return default_class 
    else:
        gainz = [information_gain(df, attribute, targetAttribute) for attribute in attributeNames]
        index_of_max = gainz.index(max(gainz)) 
        best_attr = attributeNames[index_of_max] 
        tree = {best_attr:{}}
        remainingAttributeNames = [i for i in attributeNames if i != best_attr]
        for attr_val, data_subset in df.groupby(best_attr): 
            subtree = id3(data_subset,
                          targetAttribute,                         
                          remainingAttributeNames,                         
                          default_class)
            tree[best_attr][attr_val] = subtree 
        return tree


total_entropy = entropy_of_list(df[targetAttribute]) 
print("Entropy of given Data Set:",total_entropy)
print("-----------------")

attributeNames = list(df.columns)
print("List of Attributes:", attributeNames) 
attributeNames.remove(targetAttribute) 
print("Predicting Attributes:", attributeNames)
print("-----------------")

for attribute in attributeNames:
  informationGain = str(information_gain(df, attribute, targetAttribute))
  print('Info-gain for {} is : {}'.format(attribute, informationGain))
print("-----------------")

tree = id3(df,targetAttribute ,attributeNames)
print("\n\nThe Resultant Decision Tree is :\n") 
pprint(tree)