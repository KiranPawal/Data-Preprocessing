import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("customer_churn_dataset.csv")
print(df.head())
print(df.columns)


'''----------------------------------------------------------------------------------------------------------------------------'''
'''
Line Chart
    A line chart is useful for showing a trend or ordered values.

    plt.scatter() == A scatter plot shows the relationship between two numerical columns.
'''

plt.scatter(df["tenure_months"], df["total_charges"])
plt.title("Tenure months vs Total Charges")
plt.xlabel('Tenure Months')
plt.ylabel('Total Charges')
plt.show()

''' plt.plot() == show the data points with line'''
plt.plot(df["tenure_months"], df["total_charges"])
plt.title("Tenure months vs Total Charges")
plt.xlabel('Tenure Months')
plt.ylabel('Total Charges')
plt.show()

'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''


'''
Bar Chart: Used for categorical data.
    A bar chart is useful for comparing categories.
'''
plt.bar(df['city'],df['total_charges'])
plt.xlabel("City")
plt.ylabel("Total Charges")
plt.title("City vs Total Charges")
plt.show()



'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''


'''
Histogram: A histogram shows the distribution of numerical data.
A histogram shows the distribution of numerical data. 
It tells us how frequently values occur within different ranges (called bins).
'''
plt.hist(df["total_charges"], bins=10)

plt.xlabel("Total Charges")
plt.ylabel("Frequency")
plt.title("Distribution of Total Charges")
plt.show()



'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''


'''Pie Chart: A pie chart shows parts of a whole.
A pie chart shows how a whole is divided into different categories. 
Each slice represents a category's proportion or percentage of the total.
'''
city_count = df["city"].value_counts()

plt.pie(
    city_count.values,
    labels=city_count.index,
    autopct="%1.1f%%"
)

plt.title("Customers by City")
plt.show()