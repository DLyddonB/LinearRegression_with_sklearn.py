import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#set seaborn as default style
sns.set()

from sklearn.linear_model import LinearRegression

data = pd.read_csv('1.01.+Simple+linear+regression.csv')
#print(data.head())

x = data['SAT']
y = data['GPA']
#This will return 84, which means a vector of length 84
#print(x.shape)

#this will change the dimensionality from 1D to 2D
x_matrix = x.values.reshape(-1,1)
#print(x_matrix.shape)
#
reg = LinearRegression()
#reg.fit() takes two arguments, the input [x] and the target [y]
reg.fit(x_matrix,y)
#when you run the code here, you get an error message. It is expecting a 2D array but got 1D instead
#basically our inputs are 1D objects , which sklearn doesn't like.
#we noted above that x has one single dimension and that is length. We have to reshape it into matrix

#reg.score(x,y) returns the R-squared of a linear regression
reg.score(x_matrix,y)
#print(reg.score(x_matrix,y))

#the result of reg.coef_ is an ND array containing all coefficients, here we have 0.00165.. or 0.0017
reg.coef_
#print(reg.coef_)

#this will give you the intercept
reg.intercept_
#print(reg.intercept_)

#How can we predict GPA using SAT score? reg.predict(new_inputs) returns the predictions of
# the linear regression model for some new inputs

#for example, by entering this
prediction = reg.predict([[1740]])
print(prediction)
#will give us the predicted GPA for an SAT score of 1740

new_data = pd.DataFrame(data=[1740, 1760], columns=['SAT'])
new_data_matrix = new_data.values.reshape(-1,1)
#print(new_data.values.shape)

#print(reg.fit(x_matrix,y))
