import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

pd.options.display.max_columns = 999
pd.options.display.max_rows = 999

excel_dir = r"C:\Users\dof07\PycharmProjects\bigdatalicense\practice_data\공홈자료\[Dataset] 작업형 제1유형"

mtcars = pd.read_csv(excel_dir + "\mtcars.csv")
mtcars.info()

mn = mtcars['qsec'].max()
mx = mtcars['qsec'].min()
scaled_qsec = [(i-mn)/(mx-mn) for i in mtcars['qsec']]
mtcars['scaled_qsec'] = scaled_qsec

len(mtcars[mtcars['scaled_qsec'] > 0.5])

excel_dir = r"C:\Users\dof07\PycharmProjects\bigdatalicense\practice_data\공홈자료\[Dataset] 작업형 제2유형"

X_train = pd.read_csv(excel_dir + "\X_train.csv", encoding='CP949')
X_test = pd.read_csv(excel_dir + "\X_test.csv", encoding='CP949')
y_train = pd.read_csv(excel_dir + "\y_train.csv", encoding='CP949')

# data exploration
X_train.head()
X_train.info()
X_train.describe(include = 'all')

# 범주형 변수, 수치형 변수 나누기
all_num_cols = X_train._get_numeric_data().columns
bool_cols = X_train._get_bool_data().columns
num_cols = list(set(all_num_cols) - set(bool_cols))
num_cols
non_num_cols = list(set(X_train.columns) - set(all_num_cols))
non_num_cols
X_train[non_num_cols] = X_train[non_num_cols].astype('category')
X_test[non_num_cols] = X_test[non_num_cols].astype('category')

# find NAs
X_train.isnull().sum()

# imputation for NAs
X_train['환불금액'][X_train['환불금액'].isnull()] = 0
X_test['환불금액'][X_test['환불금액'].isnull()] = 0
# find Outliers
X_train['총구매액'].sort_values(ascending = False)
X_train['최대구매액'].sort_values(ascending = False)

# Imputation for Outliers
# X_train = X_train[X_train['총구매액'] > 0]
# X_train = X_train[X_train['최대구매액'] > 0]

# Label Encoding for category variables
X_train[non_num_cols] = X_train[non_num_cols].apply(LabelEncoder().fit_transform)
X_test[non_num_cols] = X_test[non_num_cols].apply(LabelEncoder().fit_transform)

"""
Data preparation
"""
# 필요없는 칼럼 제외
X_train_id = X_train.iloc[:,0]
y_train_id = y_train.iloc[:,0]
X_test_id = X_test.iloc[:,0]

X_train = X_train.iloc[:,1:]
y_train = y_train.iloc[:,1]
X_test = X_test.iloc[:,1:]

X_train_tmp, X_val, y_train_tmp, y_val = train_test_split(X_train, y_train ,test_size=0.3)

"""
MLP
"""
mlp = MLPClassifier(hidden_layer_sizes = (30,),
                    solver = 'adam',
                    activation = 'relu',
                    learning_rate_init = 0.001,
                    max_iter = 500)
mlp.fit(X_train_tmp, y_train_tmp)

print("MLP Training Acc : ", mlp.score(X_train_tmp, y_train_tmp))
print('MLP Training ROCAUC Score: ', roc_auc_score(y_train_tmp, pd.DataFrame(mlp.predict_proba(X_train_tmp)).iloc[:,1]))

y_val7 = pd.DataFrame(mlp.predict_proba(X_val)).iloc[:,1]
y_val7 = [1 if x > 0.5 else 0 for x in y_val7 ]

print("MLP Val Acc : ", mlp.score(X_val, y_val))
print('MLP Val ROCAUC Score: ', roc_auc_score(y_val, y_val7))

cf = confusion_matrix(y_val7, y_val)
accuracy_score(y_val7, y_val)

# MLP Model Predict
predict = mlp.predict_proba(X_test)
predict = pd.DataFrame(predict).iloc[:,1]

answer = pd.concat([X_test_id, predict], axis = 1)
answer.head()
answer.to_csv()

"""
XGBOOST
"""

xgb = xgboost.XGBClassifier(n_estimators=80, learning_rate=0.03, gamma=0, subsample=0.75,
                            colsample_bytree=1, max_depth=5)
xgb.fit(X_train_tmp.iloc[:,1:],y_train_tmp.iloc[:,-1])
roc_auc_score(y_train_tmp.iloc[:,-1], pd.DataFrame(xgb.predict_proba(X_train_tmp.iloc[:,1:])).iloc[:,1])
print("XGB Training Acc : ", xgb.score(X_train_tmp.iloc[:,1:], y_train_tmp.iloc[:,-1]))
print('XGB Training ROCAUC Score: ', roc_auc_score(y_train_tmp.iloc[:,-1], pd.DataFrame(xgb.predict_proba(X_train_tmp.iloc[:,1:])).iloc[:,1]))

print("XGB Val Acc : ", xgb.score(X_val.iloc[:,1:], y_val.iloc[:,-1]))
print('XGB Val ROCAUC Score: ', roc_auc_score(y_val.iloc[:,-1], pd.DataFrame(xgb.predict_proba(X_val.iloc[:,1:])).iloc[:,1]))

# MLP Model Predict
predict = xgb.predict_proba(X_test.iloc[:,1:])
predict = pd.DataFrame(predict).iloc[:,1]

X_test_id = X_test.iloc[:,0]
answer = pd.concat([X_test_id, predict], axis = 1)
answer.to_csv()

"""
Support Vector Machine
"""
svm = SVC(C=20, gamma=1, random_state=0, probability=True)
svm.fit(X_train_tmp.iloc[:,1:],y_train_tmp.iloc[:,-1])
roc_auc_score(y_train_tmp.iloc[:,-1], pd.DataFrame(svm.predict_proba(X_train_tmp.iloc[:,1:])).iloc[:,1])
print("XGB Training Acc : ", svm.score(X_train_tmp.iloc[:,1:], y_train_tmp.iloc[:,-1]))
print('XGB Training ROCAUC Score: ', roc_auc_score(y_train_tmp.iloc[:,-1], pd.DataFrame(svm.predict_proba(X_train_tmp.iloc[:,1:])).iloc[:,1]))
print("XGB Val Acc : ", svm.score(X_val.iloc[:,1:], y_val.iloc[:,-1]))
print('XGB Val ROCAUC Score: ', roc_auc_score(y_val.iloc[:,-1], pd.DataFrame(svm.predict_proba(X_val.iloc[:,1:])).iloc[:,1]))

