import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

 # Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                        test_size = 0.25, random_state = 42) 

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#######Linear SVM Classification:

# Fitting SVM to the Training set
from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state = 0)
####RBF SVM Classification:
classifier = SVC(kernel = 'rbf', random_state = 0)
###### POLYNOMIAL########
#classifier = SVC(kernel = 'poly', random_state = 0,degree = 4)
######SIGMOID ############################
#classifier = SVC(kernel = 'sigmoid', random_state = 0)

classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy: ",accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
print("My Confusion Matrix:",confusion_matrix(y_test, y_pred))





# Visualising the Test set results







from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test 
aranged_ages = np.arange(start = X_set[:, 0].min(), 
                stop = X_set[:, 0].max(), step = 0.01)
aranged_salaries = np.arange(start = X_set[:, 1].min(),
                stop = X_set[:, 1].max(), step = 0.01)
 
X1, X2 = np.meshgrid(aranged_ages, aranged_salaries)
plt.contourf(X1, X2, classifier.predict(
    np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
     alpha = 0.5, cmap = ListedColormap(('orange', 'blue')))


for i in np.unique(y_set):
    plt.scatter(X_set[y_set == i, 0], X_set[y_set == i, 1],
                c = ListedColormap(['red', 'green'])(i), 
                label = i)

#plt.title('SVM Linear Kernel (Test set)')
plt.title('SVM RBF Kernel (Test set)')
plt.xlabel('Age')
plt.ylabel('Salary')
#plt.legend()
plt.show()
