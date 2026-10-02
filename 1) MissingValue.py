import pandas as pd 
import numpy as np

# import csv dataset
df=pd.read_csv("Titanic.csv")

# Print the columns names
print(df.columns)

# Remove the unnecessory columns in dataset
input = df[['Name', 'Sex', 'Age', 'Embarked', 'Pclass']]
target = df['Survived']

print(input.head())
print(target.head())





# It show the min(), max(), median(), mean()  values
print(df.describe())


# Print the number of column and rows
print("number of rows and columns")
print(df.shape)

'''----------------------------------------------------------------------------------------------------------------------------'''

# Print the information of all columns 
print(df.info()) 


'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''


# Find the number of missing values in each column
print(df.isnull().sum())


# Find the number of rows and columns in dataset
print(df.isnull().sum().sum())


'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''










'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''


'''Handle missing values using pandas'''

# Fill numerical columns with mean            work only with numeric column
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Fill numerical columns with median
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Fill categorical columns with mode
df["Cabin"] = df["Cabin"].fillna(df["Cabin"].mode()[0])

print(df.isnull().sum().sum())


'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''


# Method 2: Using Scikit-learn SimpleImputer
from sklearn.impute import SimpleImputer

# Handling numerical missing value 
imputer = SimpleImputer(strategy="mean")
input["Age"] = imputer.fit_transform(input[["Age"]]).ravel()

# Handling categorical missing value 
imputer = SimpleImputer(strategy="most_frequent")
input["Sex"] = imputer.fit_transform(input[["Sex"]]).ravel()











'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''


# method 3
data = pd.read_csv("messy_students_practice.csv")
data["attendance_pct"] = (
    data["attendance_pct"]
    .str.replace("%", "", regex=False)
    .replace("-", pd.NA)
    .astype("Int64")
)


# for remove the null value rows in dataset
data.dropna(inplace = True) 

# Fill the value before null value inplace of null value
data = data.bfill()

# Fill the value after null value inplace of null value
data = data.ffill()

print(data.isnull().sum())
