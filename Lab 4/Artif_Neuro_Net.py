from sklearn.neural_network import MLPClassifier
import pandas as pd
import pickle

data_heart_train = pd.read_csv("../Lab 4/Dataset/heart_train_3.csv")
data_heart_validation = pd.read_csv("../Lab 4/Dataset/heart_validation_3.csv")
X_train = data_heart_train.iloc[:,3:-1].values
Y_train = data_heart_train.iloc[:,-1].values
X_val = data_heart_validation.iloc[:,3:-1].values
y_val = data_heart_validation.iloc[:,-1].values

#Fitting the training data to the network
#Initializing the MLPClassifier

classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=100000,
activation = 'relu',solver='adam',random_state=1)
# ativations é o tipo de ativação de cada neurónio
#Solver é o tipo de treino critério de paragem para as épocas (iterações)
classifier.fit(X_train, Y_train)
print("x=",X_train[1])
print("y=",Y_train[1])
#Predicting y for X_val
filename = '../Lab 4/Dataset/predictorMLPC_heart_4.sav'
pickle.dump(classifier, open(filename, 'wb'))
print('File saved')