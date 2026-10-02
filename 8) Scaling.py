import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

df = pd.read_csv("D:\Preprocess\outliers_dataset.csv")
print(df.head())
print(df.columns)

# It show the min(), max(), median(), mean()  values
print(df.describe())



'''
1. StandardScaler
    Converts values approximately to a range centered around 0 with standard deviation 1.
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['house_id', 'area_sqft', 'bedrooms', 'age_years', 'distance_to_city_km',
       'buyer_income_k', 'price_k']])
print(df_scaled)

# create the scaled data into dataframe
data = pd.DataFrame(df_scaled, columns=[
    'house_id', 'area_sqft', 'bedrooms', 'age_years', 'distance_to_city_km',
           'buyer_income_k', 'price_k'
])

print(data.head())



'''
2. MinMaxScaler: Converts values to a range between 0 and 1.
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scale_scaled = scaler.fit_transform(df[['house_id', 'area_sqft', 'bedrooms', 'age_years', 'distance_to_city_km',
       'buyer_income_k', 'price_k']])
print(df_scale_scaled)

# create the scaled data into dataframe
scaled_data = pd.DataFrame(df_scale_scaled, columns=[
    'house_id', 'area_sqft', 'bedrooms', 'age_years', 'distance_to_city_km',
           'buyer_income_k', 'price_k'
])

print(scaled_data.head())