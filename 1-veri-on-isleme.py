########################
#### VERI ON ISLEME ####
########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

veri_yolu = "C:/Users/ramaz/Documents/ml_datas/veriler1.csv"
veriler = pd.read_csv(veri_yolu)
print(veriler.head())

# 2 farkli kolonu tek degiskene atamak
boy_kilo = veriler[['boy', 'kilo']]
print(boy_kilo.head(4))


#### 3 EKSİK VERİLER ####
veri_yolu = "C:/Users/ramaz/Documents/ml_datas/eksikveriler.csv"
veriler = pd.read_csv(veri_yolu)


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #ortalama değerleri nan değerler yerine yaz
boy_kilo_yas = veriler.iloc[:, 1:4]
print(boy_kilo_yas.head(4))

imputer = imputer.fit(boy_kilo_yas) #imputer ın yas değişkeninden öğrenmesini sağlıyoruz
boy_kilo_yas = imputer.transform(boy_kilo_yas) #.fit ile öğrendiklerini .transform ile uygulatıyoruz
print(boy_kilo_yas)
print('-'*50)


#### 4 KATEGORİK VERİLER ####
ulke = veriler.iloc[:,0:1].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0]) #tek sütunda 0 1 ve 2 olarak kodluyoruz label encoding ile OrdinalEncoder() gibi 

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray() #label encode ettikten sonra one-hot ile 0 1 li hale getiriyoruz
print(ulke)
print('-'*50)

#### 5 BİRLEŞTİRME ####
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
print(sonuc.head(4))

sonuc2 = pd.DataFrame(data=boy_kilo_yas, index=range(22), columns=['boy','kilo','yas'])
print(sonuc2.head())

cinsiyet=veriler.iloc[:,4]
sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet'])
print(sonuc3.head())

s1 = pd.concat([sonuc,sonuc2], axis=0) # axis=1 dersek satır başlı eşleştirme yapar yani birinin 1. indexini diğerinin 1. indexiyle eşler
# axis=0 dersek kolon başlı eşleştirme yapar izleyip görelim:
print(s1)

#bizim istediğimiz satır bazlı eşleştirme olsun o yüzden axis=1
s = pd.concat([sonuc,sonuc2,sonuc3], axis=1)
print(s)
print('-'*50)
s2 = pd.concat([sonuc, sonuc2], axis=1)


#### 5 TEST TRAİN BÖLME ####
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s2, sonuc3, test_size=0.3, random_state=0) # cinsiyeti tahmin edeceğimiz için bağımlı değişkenimiz cinsiyet 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train) # boy ile kilo gibi birbirinden alakasız değişkenleri kıyaslayabilmek için fit_transform kullanarak StandartScaler ile dönüştürüyoruz.
X_test = sc.fit_transform(x_test) # aynı dönüşümü teste de uyguluyoruz
print(X_train)
print(X_test)