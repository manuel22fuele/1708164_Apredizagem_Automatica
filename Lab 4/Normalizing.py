from sklearn.preprocessing import StandardScaler
import pandas as pd


# Normalizando o dataset heart_clean_1
heart_clean_1 = pd.read_csv('../Lab 4/Dataset/heart.csv')
# print(heart_clean_1)
sc = StandardScaler()
heart_clean_1[['age', 'sex', 'cp', 'trtbps','chol','fbs','exng','oldpeak','slp','caa','thall']] = \
    sc.fit_transform(heart_clean_1[['age', 'sex', 'cp', 'trtbps','chol','fbs',
                                    'exng','oldpeak','slp','caa','thall']])
heart_clean_1.to_csv("../Lab 4/Dataset/heart_clean_2.csv")
print('File saved')
