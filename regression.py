
from sklearn.datasets import load_boston
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import pandas as pd


boston = load_boston()
pd.DataFrame(boston.data).head()

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target ,test_size=0.1)
