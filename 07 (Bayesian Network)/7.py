# Program 7 (Bayesian Network)

# Install pgmpy library using this command: 
# !pip install pgmpy 

import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import PC, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

data = pd.read_csv("heart.csv")
print(data.head())

c = PC(data)
structure = c.estimate()
print(structure.edges())

model = BayesianModel(structure.edges())
model.fit(data, estimator = MaximumLikelihoodEstimator)

infer = VariableElimination(model)
q = infer.query(variables = ["cp","target"], evidence = {"sex": 0, "exang": 1})
print(q)