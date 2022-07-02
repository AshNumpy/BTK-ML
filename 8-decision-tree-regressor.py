###################
### KARAR AĞACI ###
###################

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

# ölçeklendirme
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_olcekli = ss.fit_transform(X)
y_olcekli = ss.fit_transform(Y)

# karar ağacı
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x, y)

# görselleştrime
plt.scatter(X, Y, color='orange')
plt.plot(X, r_dt.predict(X), color='blue')
plt.show()

# harici tahminler
print(r_dt.predict([[11]]))
print(r_dt.predict([[6]]))