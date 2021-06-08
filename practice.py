
import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 6]
pd.options.display.max_columns = 999
pd.options.display.max_rows = 9919

excel_dir = r"C:\Users\dof07\PycharmProjects\bigdatalicense\practice_data\내자료\midterm_practice"

### cars04
cars04 = pd.read_csv(excel_dir + "\cars04.csv")

# Data exploration (다시 해 칼럼 제대로 확인하면서)
cars04.describe(include='all')
cars04.columns
cars04.shape
cars04.head()
cars04.info()
numeric_cols_temp = cars04._get_numeric_data().columns
numeric_cols_temp
bool_cols = cars04._get_bool_data().columns
bool_cols
numeric_cols = list(set(numeric_cols_temp) - set(bool_cols))
numeric_cols
non_numeric_cols = list(set(cars04.columns) - set(numeric_cols))
non_numeric_cols

# Detect outliers for categorical data (boolean)
cat_outlier_check = [{col : cars04[col].unique()} for col in non_numeric_cols if cars04[col].dtype != 'object']
cat_outlier_check

# drop rows for integer (ncyl) outliers
cars04['ncyl'].value_counts()
cars04.drop(cars04[cars04['ncyl'] == -1].index, inplace=True)

# Data transformation : change the data type of ncyl from integer to category
cars04['ncyl'] = cars04['ncyl'].astype('category')

# Find NAs
cars04.isnull().values.any()
cars04.isnull().sum()
cars04.isnull().sum().sum()

# Imputation for NAs
cars04['city_mpg'].fillna(cars04['city_mpg'].mean() , inplace=True)
cars04['hwy_mpg'].fillna(cars04['hwy_mpg'].mean() , inplace=True)
cars04['weight'].fillna(cars04['weight'].mean() , inplace=True)
cars04['wheel_base'].fillna(cars04['wheel_base'].mean() , inplace=True)
cars04['length'].fillna(cars04['length'].mean() , inplace=True)
cars04['width'].fillna(cars04['width'].mean() , inplace=True)

# Detect outliers using box plot
plt.boxplot(cars04['city_mpg'])

# logical fields true 값들은 몇개인가

[[cars04[col].value_counts()] for col in bool_cols]

# cyl 개수 별로 무게는 얼마나 차이가 나는가. 가장 무게가 많이 나가는 cyl 값은 ?
grouped_cyl = cars04['weight'].groupby(cars04['ncyl'])
grouped_cyl.mean()
np.nanmax(grouped_cyl.mean())

# 스포츠카와 스포츠카가 아닌 것의 딜러코스트는 어떻게 차이가 나는가
grouped_sports = cars04['dealer_cost'].groupby(cars04['sports_car'])
grouped_sports.mean()

# horsepower가 가장 높은 5개의 케이스
cars04.sort_values(by=['horsepwr'], ascending=False).head(5)

# 비주얼 빈 cut & quantile의 조합(dealer cost)
dealer_labels = ['plow25pec', 'pnormal', 'phigh25pec']
pd.qcut(cars04['dealer_cost'], q=3, labels = None)
cars04['bin_dealer_cost'] = pd.qcut(cars04['dealer_cost'], q=3, labels = dealer_labels)
cars04['bin_dealer_cost'].value_counts()
cars04["dealer_cost"].groupby(cars04['bin_dealer_cost']).agg(['count','mean','std','min','max'])

dealer_labels2 = ['small', 'medium', 'large']
pd.cut(cars04['dealer_cost'], bins=3, labels = None, right = False, include_lowest=False)
cars04['bin_dealer_cost2'] = pd.cut(cars04['dealer_cost'], bins=3, labels = dealer_labels2, right = True, include_lowest=True)
cars04['bin_dealer_cost2'].value_counts()
cars04["dealer_cost"].groupby(cars04['bin_dealer_cost2']).agg(['count','mean','std','min','max'])

pd.cut(np.array(range(10)), bins=3, labels = None, right=True, include_lowest=False)
pd.cut(np.array(range(10)), bins=3, labels = None, right=False, include_lowest=False).value_counts()

pd.qcut(np.array(range(10)), q=3, labels = None)
pd.qcut(np.array(range(10)), q=3, labels = None).value_counts()


# weight가 상위 10%와 하위 10%의 가격 차이
pd.qcut(cars04['weight'], q = [0, 0.1, 0.9, 1], labels = None)
cars04['bin_weight'], bin_edges_weight = pd.qcut(cars04['weight'],
                                                 q = [0, 0.1, 0.9, 1],
                                                 labels = ["low10pec", "mid80pec", "high10pec"],
                                                 retbins = True)
cars04['bin_weight'].value_counts()
cars04['weight'].groupby(cars04['bin_weight']).agg(['count','mean','std','min','max'])
cars04['weight'].groupby(cars04['bin_weight']).mean()['high10pec'] - cars04['weight'].groupby(cars04['bin_weight']).mean()['low10pec']

# dodge만 뽑아
cars04[cars04['name'] == 'Dodge']
cars04[cars04['name'].str.contains('Dodge')]

# 브랜드별로 평균 가격은 어떻게 되는지, 내림차순으로 정렬
cars04[['brand', 'model']] = cars04['name'].str.split(" ", 1, expand = True)
cars04['dealer_cost'].groupby(cars04['brand']).mean().sort_values(ascending=False)


#### census

census = pd.read_csv(excel_dir + "\census-retail.csv")
census.head()
cols = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
excluded_cols = [col for col in census.columns if col not in cols ]
melted_census = pd.melt(census, id_vars='YEAR', value_vars= cols, var_name="MONTH", value_name="value")
melted_census.head()

#### immigration

immigration = pd.read_csv(excel_dir + "\immigration.csv")
immigration.head()
immigration.describe()
immigration.info()
immigration['party'].unique()

immigration['party'].str.split(" ", 1, expand = True)

### life
excel_dir = r"C:\Users\dof07\Desktop\빅데이터분석기사실기준비\내자료\midterm_practice"
life = pd.read_csv(excel_dir + "\life_exp_raw.csv")

life.describe()
life.head()
life.info()
life.shape

life['State'] = life['State'].str.lower()
life['County'] = life['County'].str.lower()

len(life['State'].unique())
values, counts = np.unique(life['County'], return_counts=True)
for item in zip(values, counts):
    print(item)

values2, counts2 = np.unique(life['County'], return_counts=True)
for item in zip(values2, counts2):
    print(item)

state_county = life['State'] + life['County']

values3, counts3 = np.unique(state_county, return_counts=True)
for item in zip(values3, counts3):
    print(item)

life.iloc[:,4:].describe()

student = pd.read_csv(excel_dir + "\students_with_dates.csv")
student = student.iloc[:,1:]
student.head()
student.describe()
student.info()

student[['year','month','day']] = student['dob'].str.split("-", 2, expand = True)

nurse_visit_ymdhms = student['nurse_visit'].str.split(" ", 2, expand = True)
ymd = nurse_visit_ymdhms[0].str.split("-",2,expand=True)
hms = nurse_visit_ymdhms[1].str.split(":",2,expand=True)

student['absences'].groupby(student['year']).mean().sort_values(ascending=False)

income = pd.read_csv(excel_dir + r"\us_income_raw.csv")
income.head()
income.tail()
income.describe()
income.info()
income.shape

value, counts = np.unique(income['GeoFips'], return_counts = True)
for item in zip(value, counts):
    print(item)

tf_lst = [False if len(x) > 5 else True for x in income['GeoFips']]
tf_lst
len(tf_lst)

income['GeoFips_yn'] = [False if len(x) > 5 else True for x in income['GeoFips']]
footnotes = income['GeoFips'].loc[income['GeoFips_yn']==False]
income = income.loc[income['GeoFips_yn']]
income = income.drop(['GeoFips_yn'], axis = 1)
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

#### weather
weather = pd.read_csv(excel_dir + r'\weather.csv')
weather.describe(include = 'all')
weather.info()
weather.head()
weather.isnull().sum().sum()
weather.isnull().sum()
weather = weather.drop(['X'], axis = 1)
weather_long = pd.melt(weather, id_vars = ['year', 'month', 'measure'] , var_name= 'day', value_name= 'value')
weather_long['day'] = weather_long['day'].str.replace('X', '')

pd.set_option('display.max_rows', 500)

weather_long_tidy = weather_long.loc[~weather_long['value'].isnull(),]
weather_long_tidy['day'] = pd.to_numeric(weather_long_tidy['day'])
weather_long_tidy.info()


weather_long_tidy['measure'].nunique
weather_long_tidy.groupby(['year', 'month']).count().sort_values(['year', 'month', 'day'])

weather_long_tidy.isnull().any()

weather_long

weather_long_tidy_one = weather_long_tidy[weather_long_tidy['measure'] != "Events"]

temp = weather_long_tidy[weather_long_tidy['measure'] == "Events"]
temp['value'][temp['value'] == ""].count()

weather_long_tidy['date'] = pd.to_datetime(weather_long_tidy['year'].astype(str) +'-'+ weather_long_tidy['month'].astype(str)+'-'+ weather_long_tidy['day'].astype(str))
weather_long_tidy.info()


########################################### bank

excel_dir = r"C:\Users\dof07\Desktop\빅데이터분석기사실기준비\내자료\midterm\HW1"
bank = pd.read_csv(excel_dir + r"\bank_hw.csv")
bank.head()
bank.describe()
bank.info()

# How many clients are included in the data? How many clients are younger than 30 and how many are older than 50 ?
bank.shape[0]
len(bank[bank.age < 30])
len(bank[bank.age > 50])

# "balance" field represents bank account balance in euros.
# Add new field named "balance_kw" that shows the balance in Korean won.
# Let us assume the exchange rate of currency is 1200 kw = 1 euro
bank['balance_kw'] = bank.balance * 1200

# How many clients have subscribed a term deposit? In "y" field, what is the proportion of "yes" to all clients in the data?
bank['y'].unique()
yes =  bank.y == "yes"
yes.sum()

# In "pdays" field, "-1" value means "the client was not previously contacted", change the value "-1" to NA value in the field.
# Find the number of NAs in the field.
bank.pdays == -1
pdays_null = bank["pdays"].replace({-1: None})
pdays_null.isnull().sum()

# Count the numbers of clients for each job type
bank.groupby(['job']).size()

# Add new field "age_group" that represents categorical age groups "under 20", "20~29", "30~39", "40~49", "50~59", "over 60".
# Which age group has the largest number of clients ?
bins = [0, 19, 29, 39, 49, 59, 300]
labels = ["under 20","20~29","30~39","40~49","50~59","over 60"]
bank['age_group'] = pd.cut(bank['age'], bins=bins, labels=labels)
pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199
bank[['age', 'age_group']].sort_values(by='age', ascending=False)
bank.groupby(['age_group']).size().max()

# From the "age_group" field, calculate campaign success rate for each age group (the portion of "yes" in "y" field.)
# Which age group has the highest success rate?
bank_rate = bank.groupby(['age_group'])['y'].value_counts()
bank_rate = pd.DataFrame(bank_rate)
len(bank_rate)
bank_lst = []
for i in range(0, len(bank_rate), 2):
    bank_lst.append(bank_rate[i+1] / (bank_rate[i] + bank_rate[i+1]))
bank_lst

# Calculate average contact duration ("duration" field) for each contact type ("contact" field).
bank.groupby(['contact'])['duration'].mean()

# Sort the data in ascending order of client age.
bank.sort_values(by='age', ascending=True)