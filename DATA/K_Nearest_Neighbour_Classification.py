# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                y, test_size = 0.25, random_state = 42)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(weights='distance')
classifier.fit(X_train, y_train)
    
y_pred=classifier.predict(X_test)   
    
    
from sklearn.metrics import confusion_matrix
print("My Confusion Matrix:",
      confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
print("Accuracy: ",accuracy_score(y_test, y_pred)) 
    
#new customer prediction age=28 salry=119000
k=classifier.predict(sc.transform([[28,19000]]))


print("New customer class:",k)
if k[0]==0:
    print("Not purchase type")
else:
    print("Purchase type")
    






#plotting the graph
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
aranged_ages = np.arange(start = X_set[:, 0].min(), 
                stop = X_set[:, 0].max(), step = 0.01)
aranged_salaries = np.arange(start = X_set[:, 1].min(), 
                stop = X_set[:, 1].max(), step = 0.01)


X1, X2 = np.meshgrid(aranged_ages, aranged_salaries)

plt.contourf(X1, X2, 
 classifier.predict(
     np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
     alpha = 0.7 , cmap = ListedColormap(('orange', 'blue')))


##Plotting Test set:
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())



 
plt.title('K Nearest Neighbors (Test set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

#for new customer
#plotting the graph
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
aranged_ages = np.arange(start = X_set[:, 0].min(), 
                stop = X_set[:, 0].max(), step = 0.01)
aranged_salaries = np.arange(start = X_set[:, 1].min(), 
                stop = X_set[:, 1].max(), step = 0.01)
X1, X2 = np.meshgrid(aranged_ages, aranged_salaries)

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
        X2.ravel()]).T).reshape(X1.shape),
     alpha = 0.8 , cmap = ListedColormap(('orange', 'blue')))

##Plotting Test set:
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
plt.plot(sc.transform([[28,129000]])[:,0], 
         sc.transform([[28,129000]])[:,1],'r*')
plt.title(' Customer')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()




 





 













