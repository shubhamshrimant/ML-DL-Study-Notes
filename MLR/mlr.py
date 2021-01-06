# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:05:04 2021

@author: shubh
"""
#manual encoding
#multiple linear regression

#dataset downloaded from: https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho

import numpy as np
import pandas as pd

dataset=pd.read_csv('C:\\Users\\shubh\\Downloads\\mlr\\CAR_DETAILS_FROM_CAR_DEKHO.csv')

dataset.info()

y=dataset.iloc[:,2]
data=dataset
del data['selling_price']
X=data

#as there's no null data, and name is large set we will labelencode it and other cat onehotencode

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X.iloc[:,0]=le.fit_transform(X.iloc[:,0])

#Now we will do manual encoding insteam of one hot as those categories are less
'''
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3,4,5,6])],remainder='passthrough')
X=ct.fit_transform(X)
'''

def fuel_to_num(z):
    if z=='Petrol':
        return 1
    if z=='Diesel':
        return 2
    if z=='CNG':
        return 3

def seller_type_to_num(z):
    if z=='Individual':
        return 11
    if z=='Dealer':
        return 12
    if z=='Trustmark Dealer':
        return 13
    
def transmission_to_num(z):
    if z=='Automatic':
        return 111
    if z=='Manual':
        return 112
    
def owner_to_num(z):
    if z=='First Owner':
        return 1111
    if z=='Second Owner':
        return 1112
    if z=='Third Owner':
        return 1113
    if z=='Fourth & Above Owner':
        return 1114


X['fuel']=X['fuel'].apply(fuel_to_num)
X['seller_type']=X['seller_type'].apply(seller_type_to_num)
X['transmission']=X['transmission'].apply(transmission_to_num)
X['owner']=X['owner'].apply(owner_to_num)


#splitting the data

print(X.isnull().values.any())
X['fuel']=X['fuel'].fillna((X['fuel'].mean()))  
X['seller_type']=X['seller_type'].fillna((X['seller_type'].mean()))  
X['transmission']=X['transmission'].fillna((X['transmission'].mean()))  
X['owner']=X['owner'].fillna((X['owner'].mean()))  


#X=X.fillna(X.mean(),inplace=True)
#X=np.array(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#training multiple linear regression model

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting test result

y_pred=regressor.predict(X_test)

print(y_pred)
np.set_printoptions(precision=2)
y_pred=y_pred.reshape(len(y_pred),1)
print(y_pred)

#checking for test and predict values 

#print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

#predicting for new value

data=[['Maruti Neon',2018,10200,'Petrol','Dealer','Manual','First Owner']]

data=np.array(data)
data[:,0]=le.fit_transform(data[:,0])

#data=pd.Series(data)

data[0][3]=fuel_to_num(data[0][3])
data[0][4]=seller_type_to_num(data[0][4])
data[0][5]=transmission_to_num(data[0][5])
data[0][6]=owner_to_num(data[0][6])

data=data[0]
data = data.astype(int)

#data=pd.Series(data)
        
#data=ct.fit_transform(data)

data=data.reshape(1,-1)
#data=[int(datapoint) for datapoint in data]
pred1=regressor.predict(data)

print(pred1)


#Woah!











