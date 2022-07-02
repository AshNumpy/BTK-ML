#################
### SAF BAYES ###
#################

# Sürekli değerler -> Gaussian naive bayes
# kesikli değerler -> Multinominal naive bayes
# kesikli binary değerler is -> Bernoulli naive bayes 

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

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_valid)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, gnb_pred)

true = cm[0][0] + cm[1][1]
all = sum(sum(cm))

print(y_valid.transpose().values)
print(gnb_pred.transpose())
print(cm)
print(f"GNB Başarı yüzdesi: %{(true/all)*100}")
print('*'*50)