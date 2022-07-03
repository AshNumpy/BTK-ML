### İSTENİLENLER ###
#   1-> Gerekli/Gereksiz değişkenleri bulunuz
#   2-> 5 farklı yönteme göre regresyon modeli çıkarınız (MLR, PR, SVR, DT, RFR)
#   3-> Yöntemlerin başarılarını karşılaştırınız
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm

# verileri içeri aktarma
veri_yolu = "C:/Users/ramaz/Documents/ml_datas/maaslar_yeni.csv"
veriler = pd.read_csv(veri_yolu)
print(veriler.head())
print('='*50)

#   1-> Gerekli/Gereksiz değişkenleri bulunuz
################################################################
# ML ' id kolonu alınmaz öğrenmek yerine ezberlemeye neden olur
# Unvan seviyesi olduğu için unvan kolonuna gerek yok 
# eğer unvan lvl olmasaydı label encoding yapardık unvanaa

x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]
print(x.head())
print(y.head())
X = x.values
Y = y.values


# 2-> 5 farklı yönteme göre regresyon modeli çıkarınız (MLR, PR, SVR, DT, RFR)
##############################################################################
#MLR -> Multiple Linear Regression
from sklearn.linear_model import LinearRegression
mlr_reg = LinearRegression()
mlr_reg.fit(X, Y)

model = sm.OLS(mlr_reg.predict(X), X).fit()
print('-'*32,"MULTIPLE LINEAR REGRESSION", '-'*32)
print(model.summary(),"\n"*3)


#PR -> Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
pr_reg = PolynomialFeatures(degree=2) #2. dereceden polinom
x_poly = pr_reg.fit_transform(X) # X değerlerimizi 2. dereceden polynomial dünyaya uyarlıyoruz

lin_reg_poly= LinearRegression()
lin_reg_poly.fit(x_poly, Y) #polinomal verilere göre y yi öğren

model_2 = sm.OLS(lin_reg_poly.predict(pr_reg.fit_transform(X)), X).fit()
print('-'*32,"POLINOMIAL REGRESSION", '-'*32)
print(model_2.summary(),"\n"*3)


# SVR -> Support Vector Regression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
svr_reg = SVR(kernel='rbf') # svr aykırılardan etkileniyor o yüzden veriyi ölçeklemek lazım

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc1.fit_transform(Y) #aykırı değerlerden etkilenmesin diye ölçekledik
svr_reg.fit(x_olcekli, y_olcekli)

model_3 = sm.OLS(svr_reg.predict(X), X).fit()
print('-'*32,"SUPPORT VECTOR REGRESSION", '-'*32)
print(model_3.summary(),"\n"*3)


# DT -> Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X, Y)

model_4 = sm.OLS(dt_reg.predict(X), X).fit()
print('-'*32,"DECISION TREE",'-'*32)
print(model_4.summary(),"\n"*3)


#RFR -> Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X, Y)

model_5 = sm.OLS(rf_reg.predict(X), X).fit()
print('-'*32,"RANDOM FOREST",'-'*32)
print(model_5.summary(),"\n"*3)

print('='*80)
print('-'*32,"CORRELATION",'-'*32)
print(veriler.corr())