# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:37:19 2022

@author: kawta
"""



### Importing neccessary packages
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


## Nowww loading data from pythoon !!! 
boston_dataset = datasets.load_boston()
boston_df = pd.DataFrame(boston_dataset.data)
boston_df.columns = boston_dataset.feature_names
boston_df.head()



#### Now loading data from python : Cifar10
cifar_dataset = tf.keras.datasets.cifar10.load_data()
cifar_df = pd.DataFrame(cifar_dataset.data)



### Load the dataset into Pandas Dataframe

boston_npy_target_column = np.asarray(boston_dataset.target)
boston_df['House_Price'] = pd.Series(boston_npy_target_column)


## Now separating predicators and response
predicators = boston_df.iloc[:, :-1]
response = boston_df.iloc[:, -1]
response.head()


#### Separaaaating the data iiin train and test

X_train, X_test, Y_train, Y_test = train_test_split(predicators, response, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
 

##### Nooow Let's applying a normal linead reg
linearreg = LinearRegression()
linearreg.fit(X_train, Y_train)

## prediiicting on test 
linearreg_prediction = linearreg.predict(X_test)


### let's calculate the MSE Mean squarred errrrrror
R_squared = r2_score(linearreg_prediction, Y_test)
print("R squared error on test set : ", R_squared)

## Nooow putting together the coef and their corresponding variable names
coef_df = pd.DataFrame()
coef_df["Column_Name"] = X_train.columns
coef_df["Coefficient_Value"] = pd.Series(linearreg.coef_)
print(coef_df.head(15))

plt.rcParams["figure.figsize"] = (15,6)
plt.bar(coef_df["Column_Name"], coef_df["Coefficient_Value"])


### import rrrridge regression library
from sklearn.linear_model import Ridge


#Now train the ffff model loool 

ridgeRegressor = Ridge(alpha = 0.5) ##here setting alpha 1
ridgeRegressor.fit(X_train, Y_train)
y_predicted_ridge = ridgeRegressor.predict(X_test)

### Calculating MSE
R_squared = r2_score(y_predicted_ridge, Y_test)
print("R squared error on test set : ", R_squared)


## Nooow putting together the coef and their corresponding variable names
coef_df = pd.DataFrame()
coef_df["Column_Name"] = X_train.columns
coef_df["Coefficient_Value"] = pd.Series(ridgeRegressor.coef_)
print(coef_df.head(15))



plt.rcParams["figure.figsize"] = (15,6)
plt.bar(coef_df["Column_Name"], coef_df["Coefficient_Value"])


import seaborn as sns
plt.scatter(boston_df['LSTAT'], boston_df['House_Price'])



### import Lasso regression library
from sklearn.linear_model import Lasso


#Now train the ffff model loool 

LassoRegressor = Lasso(alpha = 1) ##here setting alpha 1
LassoRegressor.fit(X_train, Y_train)
y_predicted_lasso = LassoRegressor.predict(X_test)

### Calculating MSE
R_squared = r2_score(y_predicted_lasso, Y_test)
print("R squared error on test set : ", R_squared)


## Nooow putting together the coef and their corresponding variable names
coef_df = pd.DataFrame()
coef_df["Column_Name"] = X_train.columns
coef_df["Coefficient_Value"] = pd.Series(LassoRegressor.coef_)
print(coef_df.head(15))



plt.rcParams["figure.figsize"] = (15,6)
plt.bar(coef_df["Column_Name"], coef_df["Coefficient_Value"])

























