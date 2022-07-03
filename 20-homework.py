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
print(X.isnull().sum()) # No missing value
print(X.nunique())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_valid = sc.transform(x_valid)

############
# OPERATES #
from sklearn.linear_model import LogisticRegression
log_class = LogisticRegression(random_state=0)
log_class.fit(X_train, y_train)
log_pred = log_class.predict(X_valid)

from sklearn.neighbors import KNeighborsClassifier
knn_class = KNeighborsClassifier(n_neighbors=3, metric="minkowski")
knn_class.fit(X_train, y_train)
knn_pred = knn_class.predict(X_valid)

from sklearn.svm import SVC
svc_class = SVC(kernel="poly")
svc_class.fit(X_train, y_train)
svc_pred = svc_class.predict(X_valid)

from sklearn.naive_bayes import GaussianNB
gnb_class = GaussianNB()
gnb_class.fit(X_train, y_train)
gnb_pred = gnb_class.predict(X_valid)

from sklearn.tree import DecisionTreeClassifier
dt_class = DecisionTreeClassifier(criterion="entropy")
dt_class.fit(X_train, y_train)
dt_pred = dt_class.predict(X_valid)

from sklearn.ensemble import RandomForestClassifier
rf_class = RandomForestClassifier(n_estimators=100, random_state=0)
rf_class.fit(X_train, y_train)
rf_pred = rf_class.predict(X_valid)

from sklearn.metrics import confusion_matrix
cm_log = confusion_matrix(y_valid, log_pred)
cm_knn = confusion_matrix(y_valid, knn_pred)
cm_svc = confusion_matrix(y_valid, svc_pred)
cm_gnb = confusion_matrix(y_valid, gnb_pred)
cm_dt = confusion_matrix(y_valid, dt_pred)
cm_rf = confusion_matrix(y_valid, rf_pred)

itr = 0
conf_matrixes = [cm_log, cm_knn, cm_svc, cm_gnb, cm_dt, cm_rf]
for i in conf_matrixes:
    itr += 1
    result = (i[0][0]+i[1][1]+i[2][2])/i.sum()
    print(f"{itr}. confusion matrix:\n{i}")
    print(f"Succes Rate: {round(result,3)*100}%")
    print('-'*50)