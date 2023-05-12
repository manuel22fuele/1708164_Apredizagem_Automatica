import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import pickle as p1
from sklearn.decomposition import PCA


random_state = 0

train = pd.read_csv("../Lab 3/Datasets/optdigits_tra.txt", sep=",", header=None)
test = pd.read_csv("../Lab 3/Datasets/optdigits.tes", sep=",", header=None)
train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)

overlaid_dataset = pd.concat([train, test])
X = overlaid_dataset.iloc[:,0:64]
y = overlaid_dataset.iloc[:,64]
"""print('Train Dataset: ')
print(train_df)
print('Test Dataset: ')
print(test_df)

print('Overlaid Datasets: ')
print(overlaid_dataset)"""

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=42)

pca = PCA(n_components=2) # projetamos usando PCA
pca.fit(X_train)

preditor_pca_tra = pca.fit_transform(X_train)
preditor_Pickle = open('../Lab 3/Datasets/Train_predictor_pca.csv', 'wb')
p1.dump(preditor_pca_tra, preditor_Pickle)
print('File saved')

pca.fit(X_test, y_test)
preditor_pca_tes = pca.fit(X_test, y_test)
preditor_Pickle = open('../Lab 3/Datasets/Test_predictor_pca.csv', 'wb')
p1.dump(preditor_pca_tes, preditor_Pickle)
print('File saved')


