'''
Outliers in Machine Learning — Detailed Explanation
    -An outlier is a data point that is significantly different from the majority of the data.
    -Simple example
        Suppose the ages of 10 people are:
            20, 21, 22, 20, 23, 21, 22, 24, 23, 85
        Here, 85 is an outlier because it is very far from the other values.

Causes of Outliers
    Data entry errors
    Measurement errors
    Sensor failure
    Rare events
    Genuine unusual observations

2. Types of Outliers
A. Univariate Outlier
    An unusual value in one variable.

B. Bivariate Outlier
    An observation becomes unusual when two variables are considered together.

C. Multivariate Outlier
    An observation may look normal in individual columns but become unusual when multiple features are considered together.    
'''


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

df = pd.read_csv("D:\Preprocess\outliers_dataset.csv")
print(df.head())


'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''


# how to check the outliers in the dataset
''' Detect the outlier using boxplot technique'''
numeric_columns = df.select_dtypes(include="number").columns
print(numeric_columns)

for col in numeric_columns:
    plt.figure(figsize=(6, 4))
    plt.title(col)
    sns.boxplot(x=col,data=df)
    plt.show()

print(df.shape)


'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''


''' Detect the outlier using distribution plot technique'''
'''
for column in numeric_columns:
    plt.figure(figsize=(6, 4))
    plt.title(col)
    sns.displot(df[column])
    plt.show()

    '''

'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''


# How to fixed the outliers in csv dataset:
''' Detect outliers using IQR: The IQR method is one of the easiest methods for CSV/ML datasets. '''
'''
for col in numeric_columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    IQR = q3 - q1

    min_range = q1 - 1.5*IQR
    max_range = q3 + 1.5*IQR

    outliers = df[(df[col] < min_range) | (df[col] > max_range)]

'''

'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''

 
'''Replace outliers with boundary values (Capping): Instead of deleting rows, replace extreme values with the lower/upper limits.
This is often useful when you don't want to lose rows. '''
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

''' If need to replace outliers with median:
    median = df[col].median()

    df.loc[(df[col] < lower) | (df[col] > upper), col] = median
'''


'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''

# Outlier remove using z-score method
for col in numeric_columns:
    mean = df[col].mean()
    std = df[col].std()

    z_score = (df[col] - mean) / std

    df = df[abs(z_score) < 3]


for col in numeric_columns:
    plt.figure(figsize=(6, 4))
    plt.title(col)
    sns.boxplot(x=col,data=df)
    plt.show()

print(df.shape)