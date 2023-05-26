import pandas as pd
from sklearn.model_selection import train_test_split

#Splitting the dataset into training and validation sets

heart_clean_2 = pd.read_csv("../Lab 4/Dataset/heart_clean_2.csv")
training_set, validation_set = train_test_split(heart_clean_2, test_size = 0.2,
random_state = 21)
print(type(validation_set))
training_set.to_csv("../Lab 4/Dataset/heart_train_3.csv")
validation_set.to_csv("../Lab 4/Dataset/heart_validation_3.csv")
print('Files saved')
