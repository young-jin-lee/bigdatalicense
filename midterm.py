
import pandas as pd
pd.options.display.max_columns = 999
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

help(pd.melt)

weather_long = weather_df.melt(id_vars = ['year','month','measure'], var_name = ['day'], value_name ='value')
weather_long.head()
weather_long['day']=weather_long['day'].str.replace('X', '')
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

