
import pandas as pd
pd.options.display.max_columns = 999
pd.options.display.max_rows = 999
cars_df = pd.read_csv(r"C:\Users\dof07\PycharmProjects\bigdatalicense\practice_data\내자료\midterm_practice\cars04.csv")
cars_df.info()
cars_df.head()
cars_df.shape

# 3
cars_df['name'] = cars_df['name'].astype('category')

# 4
cars_df['msrp'].mean() - cars_df['dealer_cost'].mean()

# 5
cars_df['city_mpg'].max()
top_city_mpg = cars_df.loc[cars_df['city_mpg']==60]
top_city_mpg['city_mpg']
top_city_mpg['hwy_mpg']

[cars_df['hwy_mpg'].max()]
top_hwy_mpg = cars_df.loc[cars_df['hwy_mpg']==66]
top_hwy_mpg['city_mpg']
top_hwy_mpg['hwy_mpg']

top_city_mpg['city_mpg'] - top_city_mpg['hwy_mpg']

# 6: count certain value(TRUE) in multiple columns
cars_df.iloc[:,1:6].eq(True).sum()


# 7
cars_df.loc[cars_df['suv']==True]['weight'].mean() - cars_df.loc[cars_df['minivan']==True]['weight'].mean()

# 8
cars_df['avg_mpg'] = (cars_df['city_mpg'] + cars_df['hwy_mpg']) / 2

# 9

eco_grade, bin_edges_egrade = pd.qcut(x = cars_df['avg_mpg'],
                                       q = [0, 0.2, 0.8, 1],
                                       labels = ['bad', 'normal','good'],
                                       retbins = True)
cars_df['eco_grade'] = eco_grade
bin_edges_egrade

# 10
cars_df[cars_df['all_wheel'].eq(True)]['horsepwr'].mean() - cars_df[cars_df['rear_wheel'].eq(True)]['horsepwr'].mean()
cars_df[cars_df['all_wheel'].eq(True)]['horsepwr'].describe()



weather_df = pd.read_csv(r"C:\Users\dof07\PycharmProjects\bigdatalicense\practice_data\내자료\midterm_practice\weather.csv")

weather_df.head()
weather_df.tail()
weather_df.info()
weather_df.describe()
# weather_df = weather.drop(['X'], axis = 1)
weather_df = weather_df.iloc[:, 1:]

weather_df.isna().sum()
weather_df.isnull().sum().sum()

weather_long = weather_df.melt(id_vars = ['year','month','measure'], var_name = ['day'], value_name ='value')
weather_long['day']= weather_long['day'].str.replace('X', '')
weather_long.isnull().sum()
weather_long.isnull().any()

weather_long.head()
weather_long_wide = weather_long.pivot_table(index=["year", "month", "day"], columns="measure", values = "value", aggfunc='first').reset_index()
weather_long_wide_nona = weather_long_wide.loc[~weather_long.isnull().any(axis=1)]
weather_long_wide_nona.loc[~weather_long['value'].isnull()]

weather_long_wide_nona.info()

weather_long_wide_nona['day'] = weather_long_wide_nona['day']

weather_long_wide_nona.head()
weather_long_wide_nona.describe(include = 'all')

weather_long_wide_nona['date'] = weather_long_wide_nona['year'].astype('str') +'-' + weather_long_wide_nona['month'].astype('str') + '-' +weather_long_wide_nona['day'].astype('str')
weather_long_wide_nona['date'] = pd.to_datetime(weather_long_wide_nona['date'])

weather_long_wide_nona.info()
weather_long_wide_nona = weather_long_wide_nona.drop(['year', 'month','day'], axis = 1)

weather_long_wide_nona = weather_long_wide_nona.rename_axis(None, axis=1)
weather_long_wide_nona.head()

weather_long_wide_nona = weather_long_wide_nona[~weather_long_wide_nona.isnull().any(axis = 1)]

weather_long_wide_nona['PrecipitationIn'][weather_long_wide_nona['PrecipitationIn'] == 'T'] = 0

weather_long_wide_nona.describe(include='all')
weather_long_wide_nona.head()
weather_long_wide_nona['CloudCover'] = weather_long_wide_nona['CloudCover'].astype('category')
weather_long_wide_nona['Events'] = weather_long_wide_nona['Events'].astype('object')
weather_long_wide_nona.iloc[:,2:-1] = weather_long_wide_nona.iloc[:,2:-1].astype('float')
weather_long_wide_nona.info()

weather_long_wide_nona['Max.Humidity'].describe()

weather_long_wide_nona['Max.Humidity'][weather_long_wide_nona['Max.Humidity'] == 1000] = 100

weather_long_wide_nona[weather_long_wide_nona['Events'] == ""]
weather_long_wide_nona.columns = weather_long_wide_nona.columns.str.lower()



""" 2017-2 중간고사 """

"""
1. 파일에 저장된 데이터를 data.frame으로 R에 읽어오세요. 어떤 방법으로
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

