import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as p1
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from pandas.core.common import random_state
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

data_x = [0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0]
data = data_x.split(",")
print(data)
fmap_data = map(float, data)
print(fmap_data)
flist_data = list(fmap_data)
print(flist_data)

data1 = pd.read_csv("../Lab 2/Datasets/optdigits.tes", sep=",", header= None)
data2=data1.iloc[:0,:64]
data_preparation=pd.DataFrame([flist_data],columns=list(data2))
out=data2
for x in out:
    print(x,data_preparation[x].values)
loaded_model = p1.load(open('../Lab 2/Datasets/optdigits_predictor_clf', 'rb'))
y_pred=loaded_model.predict(data_preparation)
print("preditor",int(y_pred))