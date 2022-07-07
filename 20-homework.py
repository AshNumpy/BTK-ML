import pandas as pd 
import numpy as np 
from sklearn import datasets
import warnings
warnings.filterwarnings("ignore")


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

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_valid)

from sklearn.naive_bayes import GaussianNB
gb = GaussianNB()
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_valid)

from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier()
dc.fit(X_train, y_train)
dc_pred = dc.predict(X_valid)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_valid)

################
# SUCCES RATES #
print('_'*50)
print('SUCCESS RATES:')
print("Logistic: {0:.3f}%".format(log_class.score(X_valid,y_valid)*100))
print("KNN: {0:.3f}%".format(knn_class.score(X_valid,y_valid)*100))
print("SVC: {0:.3f}%".format(svc.score(X_valid,y_valid)*100))
print("Naive Bayes: {0:.3f}%".format(gb.score(X_valid,y_valid)*100))
print("Decision Tree: {0:.3f}%".format(dc.score(X_valid,y_valid)*100))
print("Random Forest: {0:.3f}%".format(rf.score(X_valid,y_valid)*100))
print('_'*50)