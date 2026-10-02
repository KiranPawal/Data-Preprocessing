import pandas as pd 
import numpy as np

# import csv dataset
df = pd.read_csv("D:\Preprocess\messy_students_practice.csv")
print(df.head())

# handle missing values 
print(df.isnull().sum())
df.dropna(inplace=True)


'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''


# Replace one value:
df["gender"] = df["gender"].replace("M","Male")

# Replace values: Replace multiple values:
df["gender"] = df["gender"].replace({
    "M":"Male",
    "F":"Female"
})

# Replace a value in the entire DataFrame:
df = df.replace("NA","Missing")


'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''


# Check the all data type in dataset
print(df.info())


# Change data type
# String → Integer
df["student_id"] = df["student_id"].str.replace("S", "").astype(int)

# String → Float
df["fees_paid"] = df["fees_paid"].str.replace(",", "").astype(float)

print(df.head())
print(df.info())