import pandas as pd 
import numpy as np 
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:,:]
X = pd.DataFrame(X)
y = iris.target

y_names = []
for i in y:
    if i == 0:
        y_names.append("Iris-Setosa")
    if i == 1:
        y_names.append("Iris-Versicolor")
    if i == 2:
        y_names.append("Iris-Virginica")
y_names=pd.DataFrame(y_names)

#################
# PREPROCESSING #
print(type(X))
print(type(y_names))
print(X.isnull().any()) # No missing value
print(X.nunique()) # No categorical variable

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y_names)

y = pd.DataFrame(y).values
X = X.values

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(X,y, train_size=0.7, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_valid = sc.transform(x_valid)

###################
# CLASSIFICATIONS #
from sklearn.linear_model import LogisticRegression
log_class = LogisticRegression(random_state=0)
log_class.fit(X_train, y_train)
log_pred = log_class.predict(X_valid)

from sklearn.neighbors import KNeighborsClassifier
knn_class = KNeighborsClassifier(n_neighbors=3, metric="minkowski")
knn_class.fit(X_train, y_train)
knn_pred = knn_class.predict(X_valid)