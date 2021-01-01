# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 12:06:05 2021

@author: shubh
"""

#Preprocessing Part 2, taking care of nan in numerical variables, scaling, and data splitting

#importing the dataset (created in last video) and libraries

import pandas as pd
import numpy as np

dataset=pd.read_csv('op.csv')


X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

#print(X.head())
#replacing nan in numerical columns by mean

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X.iloc[:,2:4])
X.iloc[:,2:4]=imputer.transform(X.iloc[:,2:4])

#Now the missing values are replaced by the mean of other values from the column

#Now we will do encoding
#As X has independent variables, we will OneHotEncode Country Column

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

#For france the values are 100, for Spain 001, and for Germany 010

# label encoding dependent variable as it has only two values yes and no 

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)


#before this we can split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)



#Depending on the algorithm, we should choose whether to scale the data features(columns) or not.
#now we will scale. The reason is to prevent the data leakage from X_train to X_test. We splitted the data and then doing scaling
#feature scaling
from sklearn.preprocessing import StandardScaler #there are multiple methods to scale such as Standard Scaling, Min Max scaling, etc
sc=StandardScaler()
X_train[:,3:]=sc.fit_transform(X_train[:,3:])
X_test[:,3:]=sc.transform(X_test[:,3:])

#This is why we first splitted the dataset and then did fit_transform on training data and
#just transform on test set to prevent the leakage to the test set.

#Now you can use any algorithm to do your task. Preprocessing ends here for this dataset.

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)

