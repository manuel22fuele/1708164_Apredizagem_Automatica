import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("../Lab 1/Datasets/optdigits.tes", sep=",", header= None)

evaluation_data=data[:]
data_X=evaluation_data.iloc[:,0:64]
data_Y=evaluation_data.iloc[:,64]
"""print(type(evaluation_data))
print(type(data_X))"""
loaded_model = p1.load(open('../Lab 1/Datasets/optdigits_predictor', 'rb'))
print("Coefficients: \n", loaded_model.coef_)
y_pred=loaded_model.predict(data_X)
z_pred=y_pred-data_Y


right=0
wrong=0
total=0
for x in z_pred:
    z=int(x)
    total=total+1
    if z==0:
        right=right+1
    else:
        wrong=wrong+1
print('total: ', total)
print('right: ', right)
print('wrong: ', wrong)
print("Percentagens de acertos= ",right/total,"\nPercentagens de erros= ",wrong/total)