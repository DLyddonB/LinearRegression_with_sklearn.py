import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

data = pd.read_csv('1.02.+Multiple+linear+regression.csv')
print(data.head())

data.describe()
print(data.describe())

x = data[['SAT', 'Rand 1,2,3']]
y = data['GPA']

reg = LinearRegression()
reg.fit(x,y)

#
reg.coef_
print(reg.coef_)