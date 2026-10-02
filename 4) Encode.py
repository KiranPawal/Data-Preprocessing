import pandas as pd 
import numpy as np

# import csv dataset
df = pd.read_csv("categorical_data.csv")
email = pd.read_csv("Emails.csv")
print(df.head())

'''
1. Label Encoding
    -Label Encoding is a technique used to convert categorical (text) values into numerical values so that a 
    machine-learning algorithm can process them.
    -Label Encoding is particularly suitable for binary categorical variables:
        Yes / No
        Male / Female
        Pass / Fail
        True / False
'''
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])
print("\n Label Encoded Data")
print(df.head())


'''
2. One-Hot Encoding
    -This is one of the most commonly used techniques.
    -One-hot encoding creates one new column for each category:
    Here:
        1 = category is present
        0 = category is absent
    
    One-hot encoding avoids this problem:
        Pune   → [1, 0, 0]
        Mumbai → [0, 1, 0]
        Nashik → [0, 0, 1]
'''
from sklearn.preprocessing import OneHotEncoder
encoder1 = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

encoded_data = encoder1.fit_transform(df[["Education_Level"]])

encoded_df = pd.DataFrame(
    encoded_data, 
    columns=encoder1.get_feature_names_out(["Education_Level"]),
    index=df.index
    )

print("\n OneHotEncoded Data")
print(encoded_df)


'''
3. Ordinal Encoding
    -Use this when categories have a meaningful order.
    -Simple example
        Suppose we have a Satisfaction column:
        -There is a clear order:
                Very Low < Low < Medium < High < Very High
    
    -The numbers represent the order, not just different categories.
'''
from sklearn.preprocessing import OrdinalEncoder
satisfaction_order = [[
    "Low",
    "Medium",
    "High"
]]

encoder2 = OrdinalEncoder(
    categories=satisfaction_order
)

df["Satisfaction_Level"] = encoder2.fit_transform(
    df[["Satisfaction_Level"]]
)

print("\n Ordinal encoder encoded Data")
print(df["Satisfaction_Level"].head())
print(df.head())


'''
4. Frequency Encoding
    -Frequency Encoding replaces each category with its frequency.
    -Instead of creating multiple columns like One-Hot Encoding, we replace each category with its frequency or proportion.
    -One-Hot Encoding could create thousands of columns, while frequency encoding creates only one column.

    1. Simple example
        Suppose we have this City column:
            Student	    City
            1	        Pune
            2	        Mumbai
            3	        Pune
            4	        Nashik
            5	        Pune
            6	        Mumbai
            7	        Pune
            8	        Nashik

        First, count each city:
            City	Count
            Pune	4
            Mumbai	2
            Nashik	2

        Now replace the categories with their counts:
            Student	    City        Frequency
            1	        Pune        4
            2	        Mumbai      2
            3	        Pune        4
            4	        Nashik      2
            5	        Pune        4
            6	        Mumbai      2
            7	        Pune        4
            8	        Nashik      2

        So:
            Pune   → 4
            Mumbai → 2
            Nashik → 2
'''
# Calculate frequency
print("\n Frequency encoding encoded Data")
frequency = df["Payment_Method"].value_counts()
print(frequency)

df["Payment_Method"] = df["Payment_Method"].map(frequency)
print(df.head())