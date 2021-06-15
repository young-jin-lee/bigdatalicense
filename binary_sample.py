# 출력을 원하실 경우 print() 활용
# 예) print(df.head())

# 답안 제출 예시
# 수험번호.csv 생성
# DataFrame.to_csv("0000.csv", index=False)

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_auc_score, accuracy_score

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999


X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")

y_train = pd.read_csv("data/y_train.csv")

print("TRAIN: ", X_train.shape)
print("TEST: ", X_test.shape)

X = pd.concat([X_train, X_test], axis = 0)
X['환불금액'] = X['환불금액'].fillna(0)

print(pd.get_dummies(X["주구매상품"]))
print(X[['주구매상품']].apply(LabelEncoder().fit_transform))

temp = pd.get_dummies(X['주구매상품'])
X = pd.concat([X, temp], axis = 1)
X = X.drop(['주구매상품'], axis = 1)
temp = pd.get_dummies(X['주구매지점'])
X = pd.concat([X, temp], axis=1)
X = X.drop(['주구매지점'], axis=1)

X_train = X.iloc[:3500,:]
X_test = X.iloc[3500: , :]

X_train_id = X_train.iloc[:,0]
X_test_id = X_test.iloc[:,0]
y_train_id = y_train.iloc[:,0]
X_train = X_train.iloc[:,1:]
X_test = X_test.iloc[:,1:]
y_train = y_train.iloc[:,1:]


X_train_tmp, X_val, y_train_tmp, y_val = train_test_split(X_train, y_train, test_size=0.1)
print(y_train_tmp)

mlp = MLPClassifier(random_state=1, hidden_layer_sizes = (200,), solver = 'adam',activation = "relu", max_iter=400).fit(X_train_tmp, y_train_tmp.values.ravel())

y_train_tmp7 = mlp.predict_proba(X_train_tmp)
y_train_tmp7 = pd.DataFrame(y_train_tmp7).iloc[:,1]

y_val7 = mlp.predict_proba(X_val)
y_val7 = pd.DataFrame(y_val7).iloc[:,1]


print(roc_auc_score(y_train_tmp, y_train_tmp7))
y_train_tmp7 = [1 if x > 0.5 else 0 for x in y_train_tmp7]
print(accuracy_score(y_train_tmp, y_train_tmp7))

print(roc_auc_score(y_val, y_val7))
y_val7 = [1 if x > 0.5 else 0 for x in y_val7]
print(accuracy_score(y_val, y_val7))

mlp = MLPClassifier(random_state=1, hidden_layer_sizes = (200,), solver = 'adam',activation = "relu", max_iter=400).fit(X_train, y_train.values.ravel())

y_test7 = mlp.predict_proba(X_test)
y_test7 = pd.DataFrame(y_test7).iloc[:,1]
answer = pd.concat([X_test_id, y_test7], axis = 1)
answer.to_csv("abs.csv", index=False)

print(pd.read_csv("abs.csv"))














