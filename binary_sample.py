# 출력을 원하실 경우 print() 활용
# 예) print(df.head())

# 답안 제출 예시
# 수험번호.csv 생성
# DataFrame.to_csv("0000.csv", index=False)

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')
X_test = pd.read_csv('data/X_test.csv')

print(X_train.head())
print(X_train.info())
print(X_train._get_numeric_data().columns)
print(X_train._get_bool_data().columns)
print(X_train.isnull().sum())
print(X_train['환불금액'].describe())

X_train.copy()['환불금액'][X_train['환불금액'].isnull()] = 0
X_test.copy()['환불금액'][X_test['환불금액'].isnull()] = 0



X_train['주구매상품'] = X_train[['주구매상품']].apply(LabelEncoder().fit_transform)
X_train['주구매지점'] = X_train[['주구매지점']].apply(LabelEncoder().fit_transform)
X_test['주구매상품'] = X_test[['주구매상품']].apply(LabelEncoder().fit_transform)
X_test['주구매지점'] = X_test[['주구매지점']].apply(LabelEncoder().fit_transform)

X_train_id = X_train.iloc[:,0]
X_train = X_train.iloc[:,1:]
y_train_id = y_train.iloc[:,1]
y_train = y_train.iloc[:,1:]

X_test_id = X_test.iloc[:,0]
X_test = X_test.iloc[:,1:]

X_train_tmp, X_val, y_train_tmp, y_val = train_test_split(X_train, y_train, test_size = 0.3)

xgb = xgboost.XGBClassifier(n_estimators = 50, max_depth = 10, learning_rate = 0.1, gamma = 0, eval_metric = 'error', use_label_encoder=False)
xgb.fit(X_train_tmp, y_train_tmp)
train_prob = pd.DataFrame(xgb.predict_proba(X_train_tmp)).iloc[:,1] # male
y_train7 = [1 if x > 0.5 else 0 for x in train_prob]
print("XGB TRAINING AUC: ", roc_auc_score(y_train_tmp, train_prob))
print("XGB TRAINING ACC: ", accuracy_score(y_train_tmp, y_train7))

val_prob = pd.DataFrame(xgb.predict_proba(X_val)).iloc[:,1]
y_val7 = [1 if x > 0.5 else 0 for x in val_prob]
print("XGB VAL AUC: ", roc_auc_score(y_val, val_prob))
print("XGB VAL ACC: ", accuracy_score(y_val, y_val7))


xgb.fit(X_train, y_train)
y_test7 = pd.DataFrame(xgb.predict_proba(X_test)).iloc[:,1]

answer = pd.concat([X_test_id, y_test7], axis = 1)
print("ANSWER: ", answer)

answer.to_csv("0000.csv", index=False)

print("OK")
