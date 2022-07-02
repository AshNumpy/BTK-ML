#####################################
### DESTEK VEKTÖR SINIFLANDIRMASI ###
#####################################

import pandas as pd 
import numpy as np

X = pd.read_csv("veriler_X.csv", index_col = [0])
y = pd.read_csv("veriler_y.csv", index_col = [0])
print(X.head())
print(y.head())
print('*'*50)

############
# İŞLEMLER #
############

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_valid = sc.transform(x_valid)

print("'Linear' karnel tipli SVC:")
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_valid)

print(y_valid.transpose().values)
print(svc_pred.transpose())

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, svc_pred)
print(cm)

true = cm[0][0] + cm[1][1]
all = sum(sum(cm))

print(f"Doğruluk oranı %{(true/all)*100:,.2f}")
print('*'*50)

###############################################

print("'rbf' karnel tipli SVC:")
from sklearn.svm import SVC
svc2 = SVC(kernel='rbf')
svc2.fit(X_train, y_train)
svc_pred2 = svc2.predict(X_valid)

print(y_valid.transpose().values)
print(svc_pred2.transpose())

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_valid, svc_pred2)
print(cm2)

true = cm2[0][0] + cm2[1][1]
all = sum(sum(cm2))

print(f"Doğruluk oranı %{(true/all)*100:,.2f}")
print('*'*50)

###############################################