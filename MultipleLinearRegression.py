# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 23:14:15 2020

@author: Dipen Rupani
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from yellowbrick.regressor import ResidualsPlot

boston = pd.read_excel('BostonHousingData.xlsx') # loading in the data


X = pd.DataFrame(np.c_[boston['INDUS'], boston['NOX'], boston['RM'], boston['TAX'], boston['PTRATIO'],boston['LSTAT']], columns = ['INDUS', 'NOX', 'RM', 'TAX', 'PTRATIO', 'LSTAT']) # this creates the design matrix for our feature vectors
Y = boston['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2) # split data into training and testing data.

lin_model = LinearRegression() # creates a linear regression object with coeffeicients to minimize the residual sum of squares between the observed targets and and predicted targets.
lin_model.fit(X_train, Y_train) # fits data into the linear model - training the model

y_train_predict = lin_model.predict(X_train) # how the model performed on the training set.
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

y_test_predict = lin_model.predict(X_test) # how the model performed on the testing set.
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))