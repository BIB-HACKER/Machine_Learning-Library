import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression 
# Importing the dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # This line creates a matrix
y = dataset.iloc[:, 2].values 
# Fitting Linear Regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y) 

salary = lin_reg.predict([[8.3]])

print("Salary for 8.3 Label:",salary)


# Visualising the Linear Regression results
plt.plot(X, y, 'r*')

plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()