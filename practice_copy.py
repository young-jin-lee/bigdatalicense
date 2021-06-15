import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, mean_squared_error, r2_score

import warnings
warnings.filterwarnings(action='ignore', category = UserWarning)
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

excel_dir = r"C:\Users\dof07\PycharmProjects\bigdatalicense\practice_data\내자료\midterm_practice"

#### census

census = pd.read_csv(excel_dir + "\census-retail.csv")
census.head()
"""
*****************************************. 데이터프레임을 변환하여 MONTH 칼럼을 만드시오.(WIDE TO LONG)
"""

cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP',
       'OCT', 'NOV', 'DEC']
census_long = pd.melt(census, id_vars='YEAR', value_vars=cols, var_name="MONTH", value_name='VAL')
census_long.head()

#### immigration
immigration = pd.read_csv(excel_dir + "\immigration.csv")
immigration.head()
immigration.describe()
immigration.info()
"""
*. party column을 공백기준 두개의 칼럼으로 나누시오.
"""
temp = immigration['party'].str.split(" ", 1, expand=True)
immigration = pd.concat([immigration, temp], axis = 1)
immigration.drop(['party'], axis=1)
immigration.rename(columns = {0:'a', 1:'b'})


### life
excel_dir = r"C:\Users\dof07\PycharmProjects\bigdatalicense\practice_data\내자료\midterm_practice"
life = pd.read_csv(excel_dir + "\life_exp_raw.csv")

life.describe()
life.head()
life.info()
life.shape

life['State'] = life['State'].str.lower()
life['County'] = life['County'].str.lower()

len(life['State'].unique())
"""
* 각 County의 수를 구하시오
"""
life.head()
life['County'].value_counts()

"""
State와 County를 공백 기준으로 합치고 개수를 구하시오
"""
temp = life['State'] +' '+ life['County']
temp.value_counts()


### student
student = pd.read_csv(excel_dir + "\students_with_dates.csv")
student = student.iloc[:,1:]
student.head()
student.describe()
student.info()

"""
****************************************. dob를 연 월 일 세 개의 칼럼으로 나누시오
"""
temp = student['dob'].str.split("-", expand = True)
student = pd.concat([student, temp], axis = 1)
student = student.rename(columns = {0:'YEAR', 1:"MONTH", 2:"DAY"})

"""
nurse_visit을 연월일과 시간으로 나누시오
"""
temp = student['nurse_visit'].str.split(" ", expand=True)

"""
연도만 따로 추출하시오
"""
student['YEAR'] = temp[0].str.split("-", 1, expand=True)[0]

"""
*. 연도별로 결석수를 구하시오
"""
student['absences'].groupby(student['YEAR']).count().sort_values(ascending=False)

#### income

income = pd.read_csv(excel_dir + r"\us_income_raw.csv")
income.head()
income.tail()
income.describe()
income.info()
income.shape

"""
각 GeoFips의 개수를 구하시오. 
"""
income['GeoFips'].describe()
income['GeoFips'].value_counts()

"""
GeoFips_yn 칼럼을 제거하시오
"""
income = income.drop(['GeoFips_yn'], axis = 1)

"""
*****************************************. Income 칼럼에서 NA 값을 제거하고 Income 칼럼을 숫자형으로 바꾸시오
"""
income = income[~income['Income'].isnull()]
income = income[~income['Income'].isin(['(NA)'])]
income['Income'] = pd.to_numeric(income['Income'])

"""
**************************************** MAKE IT WIDE: 'Description'
**************************************** MAKE IT WIDE: 'Description'
**************************************** MAKE IT WIDE: 'Description'
"""
income.head()
income_wide = income.pivot_table(index=['GeoFips', 'GeoName', 'LineCode'], columns = 'Description', values = 'Income').reset_index()
income_wide.head()


########################################### bank

excel_dir = r"C:\Users\dof07\PycharmProjects\bigdatalicense\practice_data\내자료\midterm\HW1"
bank = pd.read_csv(excel_dir + r"\bank_hw.csv")
bank.head()
bank.describe()
bank.info()

"""
*. How many clients are included in the data? How many clients are younger than 30 and how many are older than 50 ?
"""
len(bank)
len(bank[bank['age'] < 30])
len(bank[bank['age'] > 50])

"""
# "balance" field represents bank account balance in euros.
# Add new field named "balance_kw" that shows the balance in Korean won.
# Let us assume the exchange rate of currency is 1200 kw = 1 euro
"""
bank['balance_kw'] = bank.balance * 1200
"""
# How many clients have subscribed a term deposit? In "y" field, what is the proportion of "yes" to all clients in the data?
"""
bank['y'].groupby(bank['y']).count()

"""
****************************************
# In "pdays" field, "-1" value means "the client was not previously contacted", change the value "-1" to NA value in the field.
# Find the number of NAs in the field.
"""
bank['pdays'].replace({-1:None})

bank['pdays'] = bank['pdays'].replace({-1: None})


"""
# Count the numbers of clients for each job type
"""
bank['job'].value_counts()

"""
****************************************
# Add new field "age_group" that represents categorical age groups "under 20", "20~29", "30~39", "40~49", "50~59", "over 60".
# Which age group has the largest number of clients ?
"""

bins = [0, 19,28,39,49,59,200]
labels = ["under 20", "20~29", "30~39", "40~49", "50~59", "over 60"]
bank['age_group'] = pd.cut(bank['age'], bins= bins, labels=labels)

"""
****************************************
# From the "age_group" field, calculate campaign success rate for each age group (the portion of "yes" in "y" field.)
# Which age group has the highest success rate?
"""
bank.head()
bank['y'] = bank['y'].astype('category')
rate = bank['y'].groupby(bank['age_group']).value_counts()
for i in range(0, len(rate), 2):
    print(rate[i+1] / (rate[i]+rate[i+1]))

"""
# Calculate average contact duration ("duration" field) for each contact type ("contact" field).
"""
bank['duration'].groupby(bank['contact']).mean()


"""
****************************************
# Sort the data in ascending order of client age.
"""
bank.sort_values(by = 'age')












