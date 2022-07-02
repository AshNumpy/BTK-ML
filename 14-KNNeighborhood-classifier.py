###################################
### EN YAKIN KOMŞU ALGORİTAMASI ###
###################################

import pandas as pd
import numpy as np

X = pd.read_csv("veriler_X.csv", index_col=[0]) # önceden işlemiştim verileri
y = pd.read_csv("veriler_y.csv", index_col=[0])
print(X.head())
print(y.head())
print('*'*50)

##################
# İŞLEMLER #
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_valid = sc.transform(x_valid)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_valid)

print(knn_pred)
print(y_valid.transpose())

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, knn_pred)
print("5 komşuyu alııp onlar arasından seçim yapan algoritma sonucu:")
print(cm)

true = cm[0][0] + cm[1][1]
all = sum(sum(cm))
print(f"Doğruluk oranı %{(true/all)*100:,.2f}")
print('*'*50)

######################################################################

from sklearn.neighbors import KNeighborsClassifier
knn2 = KNeighborsClassifier(n_neighbors=2, metric='minkowski')
knn2.fit(X_train, y_train)
knn_pred2 = knn2.predict(X_valid)

print(knn_pred2)
print(y_valid.transpose())

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_valid, knn_pred2)
print("2 komşuyu alıp onlar arasından seçim yapan algoritma sonucu:")
print(cm2)

true = cm2[0][0] + cm2[1][1]
all = sum(sum(cm2))
print(f"Doğruluk oranı %{(true/all)*100:,.2f}")
print('*'*50)

######################################################################