# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 11:59:07 2020

@author: shubh
"""

#Taking care of missing categorical variable.

#importing libraries
import numpy as np
import pandas as pd

#loading dataset
dataset=pd.read_csv('C:\\Users\\shubh\\Desktop\\dp.csv')

#taking out nan in categorical data 

temp_test=dataset[dataset['Country'].isnull()]

#dropping that row from the dataset

dataset=dataset.dropna()

#now building model to find the missing values
#for that doing these steps

X=dataset.iloc[:,1:4]
y=dataset.iloc[:,0]

#encoding non numeric variables
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X.iloc[:,-1]=le.fit_transform(X.iloc[:,-1])



del temp_test['Country']
temp_test.iloc[:,-1]=le.fit_transform(temp_test.iloc[:,-1])

#Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
temp_test = sc.transform(temp_test)
print(X)
print(temp_test)

#Now building the SVM model and predicting the values

from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(X,y)

#predicting a new result

print(classifier.predict(sc.transform([[50,83000,0]])))#0 as we label encoded the last column

#Now to insert into that dataset

y_pred=classifier.predict(temp_test)

#inserting this into our original dataset
#as we have deleted the row we will again import the dataset
dataset=pd.read_csv('C:\\Users\\shubh\\Desktop\\dp.csv')

dataset.loc[dataset.Country.isnull(),'Country']=y_pred

#Saving this as a csv so we will proceed further. You can do the steps in this as well.
#(Taking care of nan in numerical columns by mean using imputer.) I'll do it later

dataset.to_csv('cat_processed.csv')

ds1=pd.read_csv('cat_processed.csv')



