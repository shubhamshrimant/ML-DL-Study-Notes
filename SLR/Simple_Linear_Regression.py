# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 12:13:04 2021

@author: shubh
"""

# Simple Linear Regression on the same dataset as of previous videos

#importing libraries and dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #for visualization

dataset=pd.read_csv('op.csv')

#We will see age vs Salary 

X=dataset.iloc[:,2]
y=dataset.iloc[:,3]

#taking care of missing values 

X.fillna(X.mean(),inplace=True)
y.fillna(y.mean(),inplace=True)
#splitting the data

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#for visualization we will need X_train and X_test in original form (before scaling)

X_train1=X_train
X_test1=X_test

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=X_train.values.reshape(-1,1)
X_test=X_test.values.reshape(-1,1)
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Now we will do Simple Linear Regression

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting test set results

y_pred=regressor.predict(X_test)

#Plotting the graphs first for train data

plt.scatter(X_train1,y_train,color='red')
plt.plot(X_train1,regressor.predict(X_train),color='blue')
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

#decently fit
#plotting now for train data
plt.scatter(X_test1,y_test,color='red')
plt.plot(X_train1,regressor.predict(X_train),color='blue')
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

#not the best model but still for the concept it is okay lol.












