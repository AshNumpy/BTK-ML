##############
### R KARE ###
##############

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
print('='*50)

# verileri içeri aktarma
veri_yolu = "C:/Users/ramaz/Documents/ml_datas/maaslar.csv"
veriler = pd.read_csv(veri_yolu)
print(veriler.head())
print('='*50)

# bağımlı bağımsız seçme
x = veriler[['Egitim Seviyesi']]
y = veriler.iloc[:,2]
print(type(x))
print(type(y)) # dataframe olmalı
print('-'*50)

# df yapma
Y = pd.DataFrame(y)
X = pd.DataFrame(x)
print(type(X))
print(type(Y)) # dataframe olmalı
print('='*50)

# array yapma
X = X.values
Y = Y.values

# Random forest regressor
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
rf_reg.fit(X, Y)

# DecisionTree Regressor
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(X, Y)

# SVRegressor
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_olcekli = ss.fit_transform(X)
ss_2 = StandardScaler()
y_olcekli = np.ravel(ss_2.fit_transform(Y.reshape(-1,1)))

from sklearn.svm import SVR
sv_reg = SVR(kernel='rbf')
sv_reg.fit(x_olcekli, y_olcekli)

# R Kare
from sklearn.metrics import r2_score
rf_r_kare = r2_score(Y, rf_reg.predict(X))
dt_r_kare = r2_score(Y, dt_reg.predict(X))
sv_r_kare = r2_score(y_olcekli, sv_reg.predict(x_olcekli))

print("Random Forest R Kare: ",rf_r_kare)
print("Decision Tree R Kare: ",dt_r_kare) # dt r karesi 1 çıkar çünkü her değişken için onun değerini getiriyor
                                          # fakat farklı değişkenler için tahmin yapsa hatası çok olacak
print("Support Vector R Kare: ",sv_r_kare)