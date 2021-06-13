
import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 6]
pd.options.display.max_columns = 999
pd.options.display.max_rows = 9919

excel_dir = r"C:\Users\dof07\PycharmProjects\bigdatalicense\practice_data\내자료\midterm_practice"

#### census

census = pd.read_csv(excel_dir + "\census-retail.csv")
census.head()
"""
*. 데이터프레임을 변환하여 MONTH 칼럼을 만드시오.(WIDE TO LONG)
"""
cols = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
excluded_cols = [col for col in census.columns if col not in cols ]
melted_census = pd.melt(census, id_vars='YEAR', value_vars= cols, var_name="MONTH", value_name="value")
melted_census.head()

#### immigration
immigration = pd.read_csv(excel_dir + "\immigration.csv")
immigration.head()
immigration.describe()
immigration.info()
"""
*. party column을 공백기준 두개의 칼럼으로 나누시오.
"""
immigration['party'].unique()
temp = immigration['party'].str.split(" ", 1, expand = True)
immigration = pd.concat([pd.DataFrame(temp), immigration], axis = 1)
immigration = immigration.drop(['party'], axis=1)
immigration = immigration.rename(columns = {0:'party'})
immigration = immigration.rename(columns = {1:'aff'})
immigration.columns = ['party', 'aff', 'priority']
immigration



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
values, counts = np.unique(life['County'], return_counts=True)
for item in zip(values, counts):
    print(item)

"""
State와 County를 공백 기준으로 합치고 개수를 구하시오
"""
state_county = life['State'] + life['County']
state_county
values3, counts3 = np.unique(state_county, return_counts=True)
for item in zip(values3, counts3):
    print(item)


### student
student = pd.read_csv(excel_dir + "\students_with_dates.csv")
student = student.iloc[:,1:]
student.head()
student.describe()
student.info()

"""
*. dob를 연 월 일 세 개의 칼럼으로 나누시오
"""
student[['year','month','day']] = student['dob'].str.split("-", 2, expand = True)

"""
nurse_visit을 연월일과 시간으로 나누시오
"""
nurse_visit_ymdhms = student['nurse_visit'].str.split(" ", 2, expand = True)

"""
연도만 따로 추출하시오
"""
ymd = nurse_visit_ymdhms[0].str.split("-",2,expand=True)
student['year'] = ymd[0]

"""
*. 연도별로 결석수를 구하시오
"""
student['absences'].groupby(student['year']).mean().sort_values(ascending=False)



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
value, counts = np.unique(income['GeoFips'], return_counts = True)
for item in zip(value, counts):
    print(item)

"""
*. GeoFips에서 Footnotes를 footnotes라는 변수에 따로 저장하고 데이터프레임에서는 제거하시오
"""
income['GeoFips_yn'] = [False if len(x) > 5 else True for x in income['GeoFips']]
footnotes = income['GeoFips'].loc[income['GeoFips_yn']==False]
income = income.loc[income['GeoFips_yn']]

"""
GeoFips_yn 칼럼을 제거하시오
"""
income = income.drop(['GeoFips_yn'], axis = 1)


"""f
*. NA 값을 제거하고 Income 칼럼을 숫자형으로 바꾸시오
"""
income.info()
income.isnull().values.any()
income.isnull().sum()
income.isnull().sum().sum()

income = income.loc[~income['Income'].isin(['(NA)'])]
income['Income'] = pd.to_numeric(income['Income'])
income.head()
income

income_wide = income.pivot_table(index=["GeoFips", "GeoName", "LineCode"], columns="Description", values = "Income").reset_index()
income_wide.head()
income_wide.info()
income_wide.describe(include='all')


########################################### bank

excel_dir = r"C:\Users\dof07\PycharmProjects\bigdatalicense\practice_data\내자료\midterm\HW1"
bank = pd.read_csv(excel_dir + r"\bank_hw.csv")
bank.head()
bank.describe()
bank.info()

"""
*. How many clients are included in the data? How many clients are younger than 30 and how many are older than 50 ?
"""
bank.shape[0]
len(bank[bank.age < 30])
len(bank[bank.age > 50])
"""
# "balance" field represents bank account balance in euros.
# Add new field named "balance_kw" that shows the balance in Korean won.
# Let us assume the exchange rate of currency is 1200 kw = 1 euro
"""
bank['balance_kw'] = bank.balance * 1200
"""
# How many clients have subscribed a term deposit? In "y" field, what is the proportion of "yes" to all clients in the data?
"""
bank['y'].unique()
yes =  bank.y == "yes"
yes.sum()
"""
# In "pdays" field, "-1" value means "the client was not previously contacted", change the value "-1" to NA value in the field.
# Find the number of NAs in the field.
"""
bank.pdays == -1
pdays_null = bank["pdays"].replace({-1: None})
pdays_null.isnull().sum()
"""
# Count the numbers of clients for each job type
"""
bank.groupby(['job']).size()
"""
# Add new field "age_group" that represents categorical age groups "under 20", "20~29", "30~39", "40~49", "50~59", "over 60".
# Which age group has the largest number of clients ?
"""
bins = [0, 19, 29, 39, 49, 59, 300]
labels = ["under 20","20~29","30~39","40~49","50~59","over 60"]
bank['age_group'] = pd.cut(bank['age'], bins=bins, labels=labels)
pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199
bank[['age', 'age_group']].sort_values(by='age', ascending=False)
bank.groupby(['age_group']).size().max()
"""
# From the "age_group" field, calculate campaign success rate for each age group (the portion of "yes" in "y" field.)
# Which age group has the highest success rate?
"""
bank_rate = bank.groupby(['age_group'])['y'].value_counts()
bank_rate = pd.DataFrame(bank_rate)
len(bank_rate)
bank_lst = []
for i in range(0, len(bank_rate), 2):
    bank_lst.append(bank_rate[i+1] / (bank_rate[i] + bank_rate[i+1]))
bank_lst
"""
# Calculate average contact duration ("duration" field) for each contact type ("contact" field).
"""
bank.groupby(['contact'])['duration'].mean()
"""
# Sort the data in ascending order of client age.
"""
bank.sort_values(by='age', ascending=True)














