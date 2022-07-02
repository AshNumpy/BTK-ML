####################
### RASSAL ORMAN ###
####################

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
print('='*50)

# verileri içeri aktarma
veri_yolu = "C:/Users/ramaz/Documents/ml_datas/maaslar.csv"
veriler = pd.read_csv(veri_yolu)
print(veriler.head())
print('='*50)

# pd.dataframe olarak verileri seçme
x = veriler['Egitim Seviyesi']
y = veriler.iloc[:,2]
x = pd.DataFrame(x)
y = pd.DataFrame(y)

# np.array e dönüştürme
X = x.values
Y = y.values

# rassal orman
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0) #random_state=0 -> rastgele bir şekilde seçim yap
                                                                # n_estimators=10 ->10 tane decision tree oluşturur ve onları kullanır
rf_reg.fit(X, Y.ravel()) #.ravel() 2 boyuttan tek array'e dönüştürür

# görselleştirme
plt.scatter(X, Y, color='orange')
plt.plot(X, rf_reg.predict(X), color='blue')
plt.show()

# tahminler
print(rf_reg.predict([[8]])) # eğitim seviyesi 11 olan birinin alacağı maaş tahmini
