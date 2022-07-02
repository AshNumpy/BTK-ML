##################################
### RASSAL ORMAN SINIFLAYICISI ###
##################################

import pandas as pd 
import numpy as np 

X = pd.read_csv("veriler_X.csv", index_col=[0])
y = pd.read_csv("veriler_y.csv", index_col=[0])

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_valid = sc.transform(x_valid)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, random_state=0, criterion='entropy')
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_valid)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, rfc_pred)

print(y_valid.transpose().values)
print(rfc_pred.transpose())
print(f"RFC Başarı Yüzdesi(Standartlaştırılmış verilerle): %{((cm[0][0]+cm[1][1])/sum(sum(cm)))*100}")
print("*"*50)
