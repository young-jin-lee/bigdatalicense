
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
# temp, bin_edges_weight = pd.qcut(cars_df['weight'],
#                                                  q = [0, 0.1, 0.9, 1],
#                                                  labels = ["low10pec", "mid80pec", "high10pec"],
#                                                  retbins = True)

# 10
cars_df[cars_df['all_wheel'].eq(True)]['horsepwr'].mean() - cars_df[cars_df['rear_wheel'].eq(True)]['horsepwr'].mean()
cars_df[cars_df['all_wheel'].eq(True)]['horsepwr'].describe()