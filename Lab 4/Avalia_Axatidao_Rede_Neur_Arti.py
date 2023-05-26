from sklearn.metrics import confusion_matrix # load Confusion Matrix
import pickle
import pandas as pd

filename = "../Lab 4/Dataset/predictorMLPC_heart_4.sav"
classifier = pickle.load(open(filename, 'rb'))
data_train = pd.read_csv("../Lab 4/Dataset/heart_train_3.csv")
# As 3 primeiras colunas foram acrescentadas durante a normalização
data_validation = pd.read_csv("../Lab 4/Dataset/heart_validation_3.csv")
X_val = data_validation.iloc[:,3:-1].values
# colunas a partir da coluna 4 excepto a última
y_val = data_validation.iloc[:,-1].values # a última coluna
print(type(X_val)) #print(X_val)
y_pred = classifier.predict(X_val) # observations in y_val
cm = confusion_matrix(y_pred, y_val)
print("Traço =",cm.trace()," Matriz confusão = ",cm.sum()," score =",
classifier.score(X_val,y_val))
print("Accuracy of MLPClassifier : ", (cm))