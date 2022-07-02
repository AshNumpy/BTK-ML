############################
### POLİNOMİAL REGRESYON ###
############################
import numpy as np 
import pandas as pd 
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

# lineer regresyon
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
lin_preds = lin_reg.predict(X)


#2. derecen polinomial regresyon
from sklearn.preprocessing import PolynomialFeatures
poly_reg_1 = PolynomialFeatures(degree=2)
X_poly_1 = poly_reg_1.fit_transform(X) # X değerlerimizi 2. dereceden polynomial dünyaya uyarlıyoruz

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly_1, Y) #polinomal verilere göre y yi öğren

X_poly_trans_1 = poly_reg_1.fit_transform(X) # X_poly yi baştaki x e göre uyarlıyoruz
lin_preds_2 = lin_reg_2.predict(X_poly_trans_1)


#4. dereceden polinomial regresyon
poly_reg_2 = PolynomialFeatures(degree=4)
X_poly_2 = poly_reg_2.fit_transform(X)

lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly_2, Y)
X_poly_trans_2 = poly_reg_2.fit_transform(X)
lin_preds_3 = lin_reg_3.predict(X_poly_trans_2)


#tahminlerin gerçek değerler üzerinde görselleştirilmesi 
plt.scatter(X, Y, color='red')
plt.plot(X, lin_preds, color='green')
plt.plot(X, lin_preds_2, color='blue')
plt.plot(X, lin_preds_3, color='orange')
plt.title('Eğitim seviyesine göre maaş tahmini')
plt.show()


#MAE değerleri
model_predict_values = [lin_preds, lin_preds_2, lin_preds_3] 

def set_mae(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_true, y_pred)

index=0
for i in model_predict_values:
    index += 1
    pred = i
    print(f"{index}. Models MAE: {set_mae(Y, i)}")
print('='*50)


#Harici tahmin (Ön Kestirim)
spesific_poly_reg = poly_reg_2.fit_transform([[11]])
print("11 Edu lvls value for 3rd model: ", lin_reg_3.predict(spesific_poly_reg))
