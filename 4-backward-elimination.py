##################################
### GERİYE DOĞRU SEÇİM YÖNTEMİ ###
##################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
boy = pd.read_csv("C:/Users/ramaz/Documents/Github/boy.csv")
veri = pd.read_csv("C:/Users/ramaz/Documents/Github/veri_boysuz.csv")
print(veri.head())
print('-'*50)

import statsmodels.api as sm
X = np.append(arr=np.ones((22,1)).astype(int), values=veri, axis=1) # 22 tane 1 'den oluşan bir sütun ekleyecek veri'nin başına
X_l = veri.iloc[:,[0,1,2,3,4,5]].values # verilerin hepsinden matrix formatında X_l oluşturduk
X_l = np.array(X_l, dtype=float) # X_l yi float tipinde array e dönüştürdük
model = sm.OLS(boy, X_l).fit() # OLS(bağımlı, bağımsız) ile regresyon çıktısını aldık
print(X[0:4,:])
print('-'*50)
print(X_l[0:4,:])
print('-'*50)
print(model.summary()) # çıktıya göre önce x5 yani YAŞ değişkenini elemeliyiz anlamlı değil

# Yaş değişkeni elendi
X_l = veri.iloc[:,[0,1,2,3,5]].values #Yaşı yani 4. kolonu çıkardık
X_l = np.array(X_l, dtype=float) # X_l yi float tipinde array e dönüştürdük
model = sm.OLS(boy, X_l).fit() # OLS(bağımlı, bağımsız) ile regresyon çıktısını aldık
print('*'*50)
print(model.summary()) # geriye kalan tüm değişkenlerimizin modele katkısı anlamlı