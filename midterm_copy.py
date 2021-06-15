import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_squared_error

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

import warnings
warnings.filterwarnings(actions='ignore', category = UserWarning)

""" 2018-2 중간고사 """

cars_df = pd.read_csv(r"C:\Users\dof07\PycharmProjects\bigdatalicense\practice_data\내자료\midterm_practice\cars04.csv")
cars_df.info()
cars_df.head()
cars_df.shape


"""
*. 숫자형, 불리안, 범주형을 나누시오
"""
all_num_cols = cars_df._get_numeric_data().columns
bool_cols = cars_df._get_bool_data().columns
num_cols = list(set(all_num_cols) - set(bool_cols))
cat_cols = list(set(cars_df.columns) - set(all_num_cols))

"""
*. Detect outliers for categorical data (boolean)
"""
cat_outlier_check = [{col : cars_df[col].unique()} for col in bool_cols if cars_df[col].dtype != 'object']
cat_outlier_check

# drop rows for integer (ncyl) outliers
cars_df['ncyl'].value_counts()
cars_df.drop(cars_df[cars_df['ncyl'] == -1].index, inplace=True)
cars_df = cars_df.replace({-1:1})

# Data transformation : change the data type of ncyl from integer to category
cars_df['ncyl'] = cars_df['ncyl'].astype('category')

"""
*. Imputation for NAs
"""
cars_df['city_mpg'].fillna(cars_df['city_mpg'].mean() , inplace=True)
cars_df['hwy_mpg'].fillna(cars_df['hwy_mpg'].mean() , inplace=True)
cars_df['weight'].fillna(cars_df['weight'].mean() , inplace=True)
cars_df['wheel_base'].fillna(cars_df['wheel_base'].mean() , inplace=True)
cars_df['length'].fillna(cars_df['length'].mean() , inplace=True)
cars_df['width'].fillna(cars_df['width'].mean() , inplace=True)

cars_df['city_mpg'].fillna(cars_df['city_mpg'], inplace=True)

"""
*. logical fields true 값들은 몇개인가
"""
[[cars_df[col].value_counts()] for col in bool_cols]

"""
*. horsepower가 가장 높은 5개의 케이스
"""
cars_df.sort_values(by=['horsepwr'], ascending=False).head(5)

"""
3. name columns은 자동차 모델의 이름을 담고 있습니다. 
character와 factor type 중 어떤 것이 적절할까요? 필요하다면 type 변환을 하시오.
"""
cars_df['name'] = cars_df['name'].astype('category')
"""
4. mrsp는 소비자 권장 가격입니다. dealer_cost와 비교해서 평균적으로 차이가 얼마나 나나요?
"""
cars_df['msrp'].mean() - cars_df['dealer_cost'].mean()

"""
5. city_mpg값이 가장 큰 자동차 모델은 무엇인가요? 
그 차의 city_mpg 와 hwy_mpg의 차이는 얼마인가요? 
이 차는 hwy_mpg가 가장 높은 차와 같은 차종인가요?
"""
cars_df['city_mpg'].max()
top_city_mpg = cars_df.loc[cars_df['city_mpg']==60]
top_city_mpg['city_mpg']
top_city_mpg['hwy_mpg']
[cars_df['hwy_mpg'].max()]
top_hwy_mpg = cars_df.loc[cars_df['hwy_mpg']==66]
top_hwy_mpg['city_mpg']
top_hwy_mpg['hwy_mpg']

top_city_mpg['city_mpg'] - top_city_mpg['hwy_mpg']

"""
6. 각각의 자동차 종류 (sport car, suv, wagon, minivan, pickup)마다자동차 모델이 몇 개씩 있나요? 
어떤 종류에도 속하지 않는 자동차는 몇 개나 있나요?
"""
cars_df.iloc[:,1:6].eq(True).sum()

"""
7. SUV와 minivan 차종의 무게를 비교하시오. 평균적으로 어떤 차종이 더 무겁나요?
"""
cars_df.loc[cars_df['suv']==True]['weight'].mean() - cars_df.loc[cars_df['minivan']==True]['weight'].mean()

"""
8. 새로운 column “avg_mpg”를 추가하시오. avg_mpg는 city_mpg와 hwy_mpg의 평균입니다.
"""
cars_df['avg_mpg'] = (cars_df['city_mpg'] + cars_df['hwy_mpg']) / 2

"""
9. “eco_grade”라는 columns을 추가하시오. avg_mpg 상위 20%에는 “good” 하위 20%에는 “bad”, 나머지에는 “normal” 값을 부여하시오.
"""
eco_grade, bin_edges_grade = pd.qcut(x = cars_df['avg_mpg'],
                                       q = [0, 0.2, 0.8, 1],
                                       labels = ['bad', 'normal','good'],
                                       retbins = True)
cars_df['eco_grade'] = eco_grade
bin_edges_grade

"""
*. avg_mpg group by eco_grage
"""
cars_df['avg_mpg'].groupby(cars_df['eco_grade']).agg(['count','mean','std','min','max'])

"""
10. 4륜 구동 자동차와 후륜구동 자동차의 마력을 비교하시오
"""
cars_df[cars_df['all_wheel'].eq(True)]['horsepwr'].mean() - cars_df[cars_df['rear_wheel'].eq(True)]['horsepwr'].mean()
cars_df[cars_df['all_wheel'].eq(True)]['horsepwr'].describe()

"""
11. 브랜드별로 평균 가격은 어떻게 되는지, 내림차순으로 정렬
"""
cars_df[['brand', 'model']] = cars_df['name'].str.split(" ", 1, expand = True)
cars_df['dealer_cost'].groupby(cars_df['brand']).mean().sort_values(ascending=False)

########################################## WEATHER

weather_df = pd.read_csv(r"C:\Users\dof07\PycharmProjects\bigdatalicense\practice_data\내자료\midterm_practice\weather.csv")

"""
1. Data Exploration
"""
weather_df.head()
weather_df.tail()
weather_df.info()
weather_df.describe()
# weather_df = weather.drop(['X'], axis = 1)
weather_df = weather_df.iloc[:, 1:]
weather_df.isna().sum()
weather_df.isnull().sum().sum()

""" 
2. Tidying dataset
   - make 'day' column (wide to long)
   - make measure long to wide
"""
weather_df.columns
value_vars = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7',
       'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17',
       'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X24', 'X25', 'X26', 'X27',
       'X28', 'X29', 'X30', 'X31']
weather_long = pd.melt(weather_df, id_vars = ['year', 'month', 'measure'], value_vars = value_vars, var_name = "day", value_name = 'value')
weather_long['day'] = weather_long['day'].str.replace("X",'')
weather_long.head()

weather_long_wide = weather_long.pivot_table(index=["year", "month", "day"], columns = "measure", values= "value", aggfunc='first').reset_index()
weather_long_wide.columns.name = None


"""
4. 데이터에 year month day 세 column이 있는데 이를 하나로 합쳐서 date column을 추가하시오. 
date column은 Date data type이어야합니다. 그리고 year month day 세 column은 제거하시오
"""
weather_long_wide['date'] = weather_long_wide['year'].astype('str') + '-' + weather_long_wide['month'].astype('str') + '-' + weather_long_wide['day'].astype('str')
weather_long_wide['date'] = pd.to_datetime(weather_long_wide['date'])
weather_long_wide = weather_long_wide.drop(['year', 'month', 'day'], axis = 1)

"""
5. PrecipitationIn(강수량) 변수를 보면 “T”라는 값이 있는데 이는 Trace 비가 아주 미량왔다는 의미이다. 
해당 변수를 숫자형으로 변환할 수 있도록, “T”를 숫자 0으로 변환하시오.
"""
weather_long_wide['PrecipitationIn'] = weather_long_wide['PrecipitationIn'].replace({'T':0})

"""
7. 데이터셋에 missing values가 있나요? 몇 개나 있나요? 각 변수 별로 몇 개씩 있나요?
"""
weather_long_wide.isnull().sum()
weather_long_wide_nona = weather_long_wide.loc[~weather_long_wide.isnull().any(axis=1)]

"""
8. 각 변수의 data type을 적절한 것으로 변환하시오.
"""
weather_long_wide_nona.info()
weather_long_wide_nona.describe(include = 'all')

weather_long_wide_nona = weather_long_wide_nona.copy()
weather_long_wide_nona['CloudCover'] = weather_long_wide_nona['CloudCover'].astype('category')


weather_long_wide_nona['Events'] = weather_long_wide_nona['Events'].astype('object')
weather_long_wide_nona.iloc[:,2:-1] = weather_long_wide_nona.iloc[:,2:-1].astype('float')
weather_long_wide_nona.info()


"""
9. Max.Humidity(최대 습도) 변수를 보시오. outlier가 있나요? 
outlier 값이 실수로 0이 하나 더 붙어 나온 값이라고 합시다. 해당 outlier를 적절한 값으로 고치시오
"""
weather_long_wide_nona['Max.Humidity'].describe()
weather_long_wide_nona.shape


weather_long_wide_nona['Max.Humidity'].eq(100).count()

weather_long_wide_nona.loc[weather_long_wide_nona['Max.Humidity'] == 100, 'Max.Humidity'] = 10000

weather_long_wide_nona.loc[(weather_long_wide_nona['Max.Humidity'] == 1000), 'Max.Humidity'] = 100

weather_long_wide_nona["Max.Humidity"] = weather_long_wide_nona["Max.Humidity"].replace({1000: 1})

weather_long_wide_nona['Max.Humidity'].describe()

"""
10. Mean.VisibilityMiles(평균시야거리) 변수를 보시오. outlier가 있나요? outlier를 적절한 값으로 고치시오.
"""
weather_long_wide_nona['Mean.VisibilityMiles'].describe()
weather_long_wide_nona['Mean.VisibilityMiles'][weather_long_wide_nona['Mean.VisibilityMiles'] == 0]

"""
11. 칼럼명을 모두 소문자로 바꾸세요
"""
weather_long_wide_nona.columns = weather_long_wide_nona.columns.str.lower()




""" 2017-2 중간고사 """

"""
1. 파일에 저장된 데이터를 dataframe으로 읽어오세요. 어떤 방법으로
읽었으며 왜 그렇게 하였는지 설명하세요.

"""
insurance = pd.read_csv(r"C:\Users\dof07\PycharmProjects\bigdatalicense\practice_data\내자료\2017-2중간고사\insurance.csv")

"""
2. 읽어온 data.frame에 대해서 데이터 탐색을 수행하세요. 탐색 과정 중에
파악한 정보가 있으면 설명하고 어떻게 그것을 찾았는지 설명하세요.
"""
insurance.info()
insurance.shape
insurance.describe()


"""
3. BMI(체질량지수)는 18.5에서 24.9사이가 보통이라고 합니다. 고객을 bmi
값 (0,18.5], (18.5, 24.9], (24.9, ~) 에 따라 “light”, “normal”, “heavy” 세 그룹으로 나누어서 각
그룹에 대한 평균 보험 수령액을 비교하세요. 어떤 그룹이 가장 보험금이 많이 들었고 어떤
그룹이 적게 들었나요?
"""
help(pd.cut)
bins = [0, 18.5, 24.9, 53]
temp = pd.cut(insurance['bmi'], bins, labels = ['light', 'normal', 'heavy'])

"""
4. 남성과 여성 고객의 평균 bmi를 비교해보세요. 어떻게 다른가요? 정규화
한 bmi를 쓰는 것이 원래 주어진 bmi를 사용하는 것보다 나을까요? 왜 그렇게 생각하는지
설명을 쓰고 만약 정규화된 bmi를 사용하는 게 낫다고 생각한다면 정규화를 실제로 수행하
세요
"""
insurance['bmi'].groupby(insurance['sex']).mean()

"""
5. 보험수령액 “charges(expenses)” 변수의 값은 어떻게 분포하고 있나요? 분포에 대해
서 설명하고 왜 그런 분포가 나타나는지 설명해보세요
"""
insurance.info()
insurance["expenses"].describe()
insurance.hist(column="expenses")
insurance.boxplot(column="expenses")

"""
6. 데이터를 보았을 때, 흡연 고객이 비흡연 고객보다 보험료를 더 많이 내
야 한다고 생각하나요? 이유를 설명해보세요.
"""
insurance['expenses'].groupby(insurance['smoker']).mean()

"""
7. 데이터를 보았을 때, 더 많은 자녀가 있다면 더 많은 의료비용이 발생한
다고 볼 수 있나요? 답을 쓰고 답에 대해서 설명해보세요
"""
insurance.info()

insurance['children'].astype('category')
insurance['expenses'].groupby(insurance['children']).mean()

"""
8. 고객들 중 나이가 가장 많은 10%의 A그룹와 나이가 가장 어린 10% 그
룹 B로 나누었을 때, 그룹 A는 그룹 B에 비해서 얼마나 많은 보험 지급액이 많이 발생하나요?
"""
insurance['age_bin'], bins = pd.qcut(insurance['age'], [0, 0.1, 0.9, 1], labels = ['low10pec', 'mid80pec', 'high10pec'], retbins = True)
insurance[insurance['age_bin'] == 'low10pec']['expenses'].mean()
insurance[insurance['age_bin'] == 'high10pec']['expenses'].mean()

"""
9. 성별에 따른 평균 보험지급액을 비교해보세요? 어떻게 다른가요?
"""
insurance['expenses'].groupby(insurance['sex']).mean()

"""
10. 남성과 여성을 비교했을 때 흡연 고객의 비율은 어디가 더 많은가요? 만
약 남성과 여성의 보험지급액이 다르다면 그것은 두 성별간의 흡연 고객의 비율이 다르기 때
문이라고 설명할 수 있을까요? 그렇지 않다면 흡연 여부와 상관없이 성별의 차이 때문에 발
생한다고 할 수 있을까요? 이것에 답하기 위한 정보를 데이터로부터 찾아보고 답을 해보세요
"""

temp = insurance.groupby(['sex', 'smoker'])['expenses'].describe()
temp
temp['count']

insurance.groupby(['smoker'])['expenses'].describe()

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
insurance.head()
insurance.info()

#insurance.loc[:,['sex','smoker', 'region', 'age_bin']] = insurance.loc[:,['sex','smoker','region','age_bin']].apply(LabelEncoder().fit_transform)

insurance.head()

children_dum = pd.get_dummies(insurance['children'], prefix = 'children')
region_dum = pd.get_dummies(insurance['region'], prefix = 'region')
pd.get_dummies(insurance['age_bin'], prefix = 'age_bin')
male_dum = pd.get_dummies(insurance['sex'], prefix = 'sex')['sex_male']
smoker_dum = pd.get_dummies(insurance['smoker'], prefix = 'smoker')['smoker_yes']
help(pd.concat)

insurance.head()
insurance[['age', 'bmi', 'smoker','expenses']]
insurance_ready = pd.concat([insurance[['age', 'bmi', 'expenses']], pd.DataFrame(children_dum), pd.DataFrame(region_dum), pd.DataFrame(male_dum), pd.DataFrame(smoker_dum)], axis = 1)
insurance_ready.columns
X = insurance_ready[['age', 'bmi', 'children_0', 'children_1', 'children_2',
       'children_3', 'children_4', 'children_5', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest', 'sex_male',
       'smoker_yes']]
y = insurance_ready['expenses']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state=123)
Xtrain.shape
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm
from statsmodels.formula.api import ols


help(LinearRegression)
lireg_train = LinearRegression().fit(Xtrain, ytrain)
lireg_train.score(X,y)
lireg_train.coef_

lireg_train.intercept_
y_hat_test = lireg_train.predict(Xtest)

y_hat_test[:6]
ytest.head()
mean_squared_error(y_hat_test, ytest)
r2_score(y_hat_test, ytest)

lireg_train.summary
model = sm.OLS(ytrain, Xtrain)
insurance_ready.columns

model = ols('expenses~' + "+".join(insurance_ready.columns), data = insurance_ready).fit()

model.summary()
