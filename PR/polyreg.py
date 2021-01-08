# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 18:42:10 2021

@author: shubh
"""

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:\\Users\\shubh\\Downloads\\position_salaries.csv')
dataset.info()
dataset.dropna(inplace=True)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values



from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

X_test1=poly_reg.transform(X_test)
y_pred=lin_reg_2.predict(X_test1)
print(y_pred)

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
