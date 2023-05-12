
import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score



data_x = [0,1,6,15,12,1,0,0,0,7,16,6,6,10,0,0,0,8,16,2,0,11,2,0,0,5,16,3,0,5,7,0,0,7,13,3,0,8,7,0,0,4,12,0,1,13,5,0,0,0,14,9,15,9,0,0,0,0,6,14,7,1,0,0]
#data_x=int(input("introduza valores do dígito: "))
data=data_x
print(data)
fmap_data = map(float, data)
print(fmap_data)
flist_data = list(fmap_data)
print(flist_data)
data1 = pd.read_csv("../Lab 1/Datasets/optdigits_tra.txt", sep=",")
data2=data1.iloc[:0,:64]
data_preparation=pd.DataFrame([flist_data],columns=list(data2))
out=data2
for x in out:
    print(x, data_preparation[x].values)
loaded_model = p1.load(open('../Lab 1/Datasets/optdigits_predictor', 'rb'))
y_pred=loaded_model.predict(data_preparation)
print("wine quality",int(y_pred))
