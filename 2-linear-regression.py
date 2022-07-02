################################
### BASİT DOĞRUSAL REGRESYON ###
################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import
veri_yolu = "C:/Users/ramaz/Documents/ml_datas/satislar.csv"
veriler = pd.read_csv(veri_yolu)
print(veriler.head())

#set dependent and independent variables
aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]

#set train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.3, random_state=0)

print(x_train.head())
print(x_test.head())

# set independent value to transform for understandable 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
print(X_train)
print(X_test) # bunları öylesine yazdım burada 1 tane bağımsızımız olduğu için transforma gerek yok aslında
# ama birden çok bağımsızımız olsaydı onları transform etmek mantıklı olurdu modelin kıyaslayabilmesi açısından

# doğrusal regresyon model inşaası
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
preds = lr.predict(x_test)

# verileri görselleştirme
x_train = x_train.sort_index() #indexe göre sort et
y_train = y_train.sort_index() #indexe göre y_traini sort et

plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))
plt.title('Aylara Göre Satış')
plt.xlabel('Aylar')
plt.ylabel('Satışlar')
plt.show()