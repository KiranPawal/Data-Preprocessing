import pandas as pd 
import numpy as np

# import csv dataset
df = pd.read_csv("Titanic.csv")
print(df.head())


input = df[['Pclass','Sex','Age','Fare']]
target = df['Survived']
print(input.head())
print(target.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(input,target,test_size=0.2,random_state=21)
print(x_train.shape)
print(x_test.shape)