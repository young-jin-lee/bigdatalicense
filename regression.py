from sklearn.datasets import load_boston



import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

data = load_boston() # Loading the data

X = pd.DataFrame(data.data, columns=data.feature_names) # Feature matrix in pd.DataFrame format
y = pd.Series(data.target) # Target vector in pd.Series format

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
xgb = xgboost.XGBRegressor(max_depth=3, n_estimators = 100, n_jobs=2, objective='reg:squarederror',
                           booster='gbtree', random_state=42, learning_rate=0.05)
xgb.fit(X_train, y_train)
y_train7 = xgb.predict(X_train)
y_test7 = xgb.predict(X_test)
train_mse = mse(y_train, y_train7)
train_rmse = np.sqrt(train_mse)
train_rmse
test_mse = mse(y_test, y_test7)
test_rmse = np.sqrt(test_mse)
test_rmse

r2_score(y_test, y_test7)