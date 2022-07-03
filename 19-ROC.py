##############################
### ALICI ÇALIŞMA ÖZELLİĞİ ###
##############################

import pandas as pd
import numpy as np 
import matplotlib.pyplot as pyplot

X = pd.read_csv("veriler_X.csv", index_col=[0])
y = pd.read_csv("veriler_y.csv", index_col=[0])

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(X,y, train_size=0.7, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_valid = sc.transform(x_valid)

print("|--------C2--------|--------C2--------|")
print("|--True Positive---|--False Negative--|")
print("|--False Negative--|--True Positive---|")
print('-'*50)

print("|__________________|____Predicted Yes____|_____Predicted No____|")
print("|_____Real Yes_____|__________X__________|__________Y__________|")
print("|_____Real No______|__________Z__________|__________W__________|")

print("TPR -> Gerçekte doğru olanların kaçı doğru tahmin edilmiş")
print("FPR -> Gerçekte yanlış olanların kaçı doğru tahmin edilmiş")
print("TPR = [X/(X+Y)]")
print("FPR = [Z/(Z+W)]")
print('-'*50)

#####################################################
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_valid)

from sklearn.metrics import confusion_matrix
cm_log = confusion_matrix(y_valid, log_reg_pred)

TPR_log = cm_log[0][0]/(cm_log[0][0]+cm_log[0][1])
FPR_log = cm_log[1][0]/(cm_log[1][0]+cm_log[1][1])

#####################################################
from sklearn.svm import SVC
svc2 = SVC(kernel='rbf')
svc2.fit(X_train, y_train)
svc_pred2 = svc2.predict(X_valid)

print(y_valid.transpose().values)
print(svc_pred2.transpose())

from sklearn.metrics import confusion_matrix
cm_svc = confusion_matrix(y_valid, svc_pred2)

TPR_svc = cm_svc[0][0]/(cm_svc[0][0]+cm_svc[0][1])
FPR_svc = cm_svc[1][0]/(cm_svc[1][0]+cm_svc[1][1])

#####################################################
