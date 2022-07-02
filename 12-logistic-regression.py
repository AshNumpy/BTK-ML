##########################
### LOJİSTİK REGRESYON ###
##########################

import pandas as pd 
import numpy as np 

veri_yolu = "C:/Users/ramaz/Documents/ml_datas/eksikveriler.csv"
veriler = pd.read_csv(veri_yolu)
print(veriler)
print('*'*50)

######################
### VERİ ÖN İŞLEME ###
ulke = veriler.iloc[:,0:1].values
cinsiyet = veriler.iloc[:,4:].values
boy_kilo= veriler.iloc[:,1:3]
yas = veriler.iloc[3,:]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
cinsiyet[:,0] = le2.fit_transform(veriler.iloc[:,-1])

ohe = preprocessing.OneHotEncoder()
ohe2 = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
cinsiyet = ohe.fit_transform(cinsiyet).toarray()

ulke = pd.DataFrame(data=ulke, index=range(len(veriler)), columns=["fr","tr","us"])

cinsiyet = pd.DataFrame(data=cinsiyet, index=range(len(veriler)))

X = pd.concat([ulke,boy_kilo], axis=1)
y = cinsiyet.iloc[:,0]

X.to_csv("veriler_X.csv", header=True)
y.to_csv("veriler_y.csv", header=True)

print(X.head(3))
print(y.head(3))
print('*'*50)

################
### İŞLEMLER ###
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(X,y, train_size=0.7, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_valid = sc.transform(x_valid)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

log_reg_pred = log_reg.predict(X_valid)

print(x_valid)
print(y_valid)
print(log_reg_pred)
