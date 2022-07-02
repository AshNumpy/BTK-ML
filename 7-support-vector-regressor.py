##################################
### DEESTEK VEKTÖRÜ REGRESYONU ###
##################################

# aykırı değerlere karşı çok hassas bir algoritme
# o yüzden aykırı değer temizlği veya replace i lazım
# StandartScaler lazım

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
print('='*50)

# verileri içeri aktarma
veri_yolu = "C:/Users/ramaz/Documents/ml_datas/maaslar.csv"
veriler = pd.read_csv(veri_yolu)
print(veriler.head())
print('='*50)

# dataframe olarak seç
x = veriler['Egitim Seviyesi']
y = veriler.iloc[:,2]
x = pd.DataFrame(x)
y = pd.DataFrame(y)

# np.array 'e dönüştür
X = x.values
Y = y.values

# Verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc1.fit_transform(y) #aykırı değerlerden etkilenmesin diye ölçekledik

# SVRegresyon
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli, color='orange')
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color='black')
plt.show()

# harici tahmin
print(svr_reg.predict([[11]]))