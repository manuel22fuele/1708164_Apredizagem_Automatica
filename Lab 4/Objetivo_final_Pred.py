import pickle
import pandas as pd

filename ="../Lab 4/Dataset/predictorMLPC_heart_4.sav"
classifier = pickle.load(open(filename, 'rb'))
val_input=input("Introduce a value: ")
val=val_input.split(",")
val0=[['age', 'sex', 'cp', 'trtbps','chol','fbs','exng','oldpeak','slp','caa','thall']]

list1=[]

for u in range(0,len(val0)):
    list1.append(val[u])

df = pd.DataFrame([list1],columns=[['age', 'sex', 'cp', 'trtbps','chol',
                                        'fbs','exng','oldpeak','slp','caa','thall']])
val_pred = classifier.predict(df.values)
print(val_pred)

