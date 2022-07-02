import pandas as pd 
import numpy as np 

veri_yolu = "C:/Users/ramaz/Documents/ml_datas/odev_tenis.csv"
veriler = pd.read_csv(veri_yolu)
print(veriler.head())
print("Verilerin satır uzunluğu: ", len(veriler))
print('='*50)


#ENCODING: CATEGORICAL -> NUMERICAL
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

#outlook
outlook = veriler.iloc[:,0:1].values
outlook[:,0] = le.fit_transform(veriler.iloc[:,0])
outlook = ohe.fit_transform(outlook).toarray()
print(outlook)
print('='*50)
outlook = pd.DataFrame(outlook, index=range(len(veriler)), columns=['overcast', 'rainy', 'sunny'])

#windy
windy = veriler.iloc[:,3:4].values
windy[:,0] = le.fit_transform(veriler.iloc[:,3:4])
windy = ohe.fit_transform(windy).toarray()
windy = pd.DataFrame(data=windy[:,1], columns=['windy']) #one-hot encodingten tekli binary seçimi
print(windy)
print('='*50)
windy = pd.DataFrame(windy, index=range(len(veriler)), columns=['windy'])

#play
play = veriler.iloc[:,-1]
play = pd.DataFrame(play)
play = play.values
play[:,0] = le.fit_transform(veriler.iloc[:,-1])
play = ohe.fit_transform(play).toarray()
play = pd.DataFrame(data = play[:,1], index=range(len(veriler)), columns=['play']) # `no` seçeneğini 0 `yes`kısmını 1 almak istediğim için 1. kolonu seçtim 
print(type(play))
print(play)


#CONCAT
others = veriler[['temperature', 'humidity']]
sonuc = pd.concat([outlook, others, windy, play], axis=1)
print(sonuc.head())
print('=',50)
print(veriler.head())

# SPLIT DATA
y = sonuc['play']
x = sonuc.iloc[:,:4]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# MODEL
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
regressor = LinearRegression()
regressor.fit(x_train, y_train)
preds = regressor.predict(x_test)
preds = np.round_(preds) # tahminlerimiz ya 0 ya 1 olması gerektiği için çıkan verilerimizi yuvarladık ki tam sayı çıksın

print('='*50)
print(f"Predictions MAE: {mean_absolute_error(y_test, preds)}") # %40 hatalı tahminler yapmışız
print('='*50)
print(y_test)
print(preds)
print('='*50)

# MODEL İYİLEŞTİRME
import statsmodels.api as sm
#1
backward_data_x_values = sonuc.iloc[:,[0,1,2,3,4,5]].values # baştaki x verilerinin formatını np.ndarray yaptık
backward_data_x_values = np.array(backward_data_x_values, dtype=float) # backward_data_x_values 'i float tipinde array e dönüştürdük
reg_model = sm.OLS(y, backward_data_x_values).fit() # OLS(bağımlı, bağımsız) ile regresyon çıktısını aldık
print(reg_model.summary()) # çıktıya göre x4 değişkeni modelden çıkartılmalı

#2
backward_data_x_values = sonuc.iloc[:,[0,1,2,4,5]].values
backward_data_x_values = np.array(backward_data_x_values, dtype=float)
reg_model_2 = sm.OLS(y, backward_data_x_values).fit()
print(reg_model_2.summary()) # 4. değer çıkıyor

#3
backward_data_x_values = sonuc.iloc[:,[0,1,2,5]].values
backward_data_x_values = np.array(backward_data_x_values, dtype=float)
reg_model_2 = sm.OLS(y, backward_data_x_values).fit()
print(reg_model_2.summary()) # 4. değer çıkıyor

#4
backward_data_x_values = sonuc.iloc[:,[0,1,2]].values
backward_data_x_values = np.array(backward_data_x_values, dtype=float)
reg_model_2 = sm.OLS(y, backward_data_x_values).fit()
print(reg_model_2.summary()) # tüm değişkenler anlamlı

#final model
son_model = sonuc.iloc[:,[0,1,2]].values
r_encoder = LinearRegression()
r_encoder.fit(son_model, y)
final_preds = r_encoder.predict(x_test.iloc[:,[0,1,2]])
final_preds = np.round_(final_preds)
print('='*50)
print(y_test)
print(final_preds)
print(f'MAE Final Predictions: {mean_absolute_error(y_test, final_preds)}')