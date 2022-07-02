###################################
### KARAR AĞACI SINIFLANDIRMASI ###
###################################

import pandas as pd 
import numpy as np 

X = pd.read_csv("veriler_X.csv", index_col=[0])
y = pd.read_csv("veriler_y.csv", index_col=[0])

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_valid = sc.fit_transform(x_valid)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy') # entropy tipinde sınıflandırma
dtc.fit(X_train, y_train)
dtc_pred = dtc.predict(X_valid)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, dtc_pred)

print(y_valid.transpose().values)
print(dtc_pred.transpose())
print(f"DTC Başarı Oranı (Standartlaştırılmış verilerle): %{((cm[0][0]+cm[1][1])/sum(sum(cm)))*100}")
print('*'*50)

#####################################################################################################

from sklearn.tree import DecisionTreeClassifier
dtc2 = DecisionTreeClassifier(criterion='entropy')
dtc2.fit(x_train, y_train)
dtc_pred2 = dtc2.predict(x_valid)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_valid, dtc_pred2)

print(y_valid.transpose().values)
print(dtc_pred2.transpose())
print(f"DTC Başarı Oranı (normal verilerle): %{((cm2[0][0]+cm2[1][1])/sum(sum(cm2)))*100}")