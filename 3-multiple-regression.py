################################
### ÇOKLU DOĞRUSAL REGRESYON ###
################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# veri ön işleme
veri_yolu = "C:/Users/ramaz/Documents/ml_datas/veriler.csv"
veriler = pd.read_csv(veri_yolu)
print(veriler.head())

c = veriler[['cinsiyet']] #-> kolon pandas Data Frame tipinde olmalı
c = c.values #-> data frame den aray şekline getiriyoruz.
print(c)
print(type(c))


# CINSIYET ENCODING: -> CATEGORICAL TO NUMERIC
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
c[:,0] = le.fit_transform(veriler[['cinsiyet']]) # -> cinsiyeti ordinal sıraya soktuk 0 1 2 diye ama 2 tür olduğu için 0 1 de kaldı yanlış anlaşılmasın

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray() # -> array şeklinde 0 1 li sütunlara dönüştürdük
print(c)
# -> kukla değişken için n-1 kadarını alacağız regresyondan hatırla

sonuc3 = pd.DataFrame(data=c[:,:1], index=range(22), columns=['cinsiyet']) #-> bizim c değerimiz 2 sütunlu olduğu için 1 sini aldık o da baştaki olsun
print(sonuc3)


# ULKE ENCODING: -> CATEGORICAL TO NUMERIC
ulke = veriler.iloc[:,0:1].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0]) #tek sütunda 0 1 ve 2 olarak kodluyoruz label encoding ile OrdinalEncoder() gibi 

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray() #label encode ettikten sonra one-hot ile 0 1 li hale getiriyoruz
print(ulke)
print('-'*50)

sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr', 'tr', 'us'])
print(sonuc.head())


# CONCAT DATAS
boy_kilo_yas = veriler.iloc[:,1:4]
print(boy_kilo_yas.head()) 

s = pd.concat([sonuc, boy_kilo_yas, sonuc3], axis=1) # -> aixs=1 satır bazlı birleştirme axis=0 sütun bazlı eşleştirme
print(s.head())

s2 = pd.concat([sonuc, boy_kilo_yas], axis=1)


# TEST TRAIN SPLIT
from sklearn.model_selection import train_test_split
x = s2
y = sonuc3
print('-'*50)
print(x.head())
print(y.head())
print('-'*50)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# MULTIPLE REGRESSION
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(f"MAE predicts: {mean_absolute_error(y_test, y_pred)}")
print('-'*50)


# BOYU TAHMİN ETME
boy = veriler[['boy']]
boy.to_csv('boy.csv', encoding='utf-8', index = False)

boy = veriler[['boy']].values

sol = s.iloc[:,:3] # -> boy un sol tarafı 
sağ = s.iloc[:,4:] # -> boy un sağ tarafı

veri = pd.concat([sol, sağ], axis=1)
veri.to_csv('veri_boysuz.csv', encoding='utf-8', index=False)
print(veri.head())

x_train, t_test, y_train, y_test = train_test_split(veri, boy, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
r2 = LinearRegression()
r2.fit(x_train, y_train)
y_pred = r2.predict(x_test)
print(y_pred)
print(y_test)
