import numpy as np
import pandas as pd 
# Importing the dataset
dataset = pd.read_csv('Employee_Data.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
print(X)
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
print("##########After Label Encoding###########")
print(X)
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, 
                    test_size = .25, random_state = 42)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred=regressor.predict(X_test)


from sklearn import metrics
print("Error:",np.sqrt(metrics.mean_squared_error
                       (y_test,y_pred)))
print("C=",regressor.intercept_)
m=regressor.coef_
print("m=",m)

print("result2",
      regressor.predict([[1, 2700, 0, 2.3]]))

accuracy = (regressor.score(X_test,y_pred))
print("Accuracy=",accuracy)


















