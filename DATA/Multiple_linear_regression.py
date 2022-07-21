import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importing the dataset
dataset = pd.read_csv('Employee_Data.csv')
X = dataset.iloc[:, :4].values
y = dataset.iloc[:, 4].values
print("########Features##################")
print(X)
print("##############Output#####################")
print(y)

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
la = LabelEncoder()
K = la.fit_transform(X[:, 0])


X[:, 0]=K
print("#After updation ####################")
print(X)

oneh = OneHotEncoder(categories='auto')
X = oneh.fit_transform(X).toarray()
print("###########after one hot encoding######")
print(X)
print("############################")
X=X[:,1:]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, 
                            test_size = 1/3, random_state = 0) 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred=regressor.predict(X_test)
# print the coefficients
accuracy = (regressor.score(X_test,y_pred))
print("Accuracy=",accuracy)









