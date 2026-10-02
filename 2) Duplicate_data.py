import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

df = pd.read_csv("D:\Preprocess\duplicate_practice_orders.csv")
print(df.head())
print(df.columns)


# 2. Count duplicate rows
print("\nCount duplicate rows")
print(df.duplicated().sum())


# Display duplicate rows
print("\nDisplay duplicate rows")
print(df[df.duplicated()])


# To show all rows involved in duplicates:
print("\nTo show all rows involved in duplicates")
print(df[df.duplicated(keep=False)])


'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''


# Remove duplicate rows
print("\nRemove duplicate rows")
df = df.drop_duplicates()


# Remove duplicates based on a specific column
df = df.drop_duplicates(subset="email")


# Keep the last occurrence
print("\nKeep the last occurrence")
df = df.drop_duplicates(keep="last")


# Count duplicate rows
print("\nCount duplicate rows")
print(df.duplicated().sum())