import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("../Lab 1/Datasets/optdigits_tra.txt", sep=",", header=None)


train_data=data[:]
data_X=train_data.iloc[:,0:64]
data_Y=train_data.iloc[:,64]
#print(train_data.columns)
print(data_X)
print(data_Y)
regr = linear_model.LinearRegression()
preditor_linear_model=regr.fit(data_X, data_Y)
preditor_Pickle = open('../Lab 1/Datasets/optdigits_predictor', 'wb')
print("Lab 1/Datasets/optdigits_predictor")
p1.dump(preditor_linear_model, preditor_Pickle)
