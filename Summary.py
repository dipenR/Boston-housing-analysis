# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 23:28:00 2020

@author: Dipen Rupani
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

boston = pd.read_excel('BostonHousingData.xlsx') # loading in the data

X = pd.DataFrame(np.c_[boston['INDUS'], boston['NOX'], boston['RM'], boston['TAX'], boston['PTRATIO'],boston['LSTAT']], columns = ['INDUS', 'NOX', 'RM', 'TAX', 'PTRATIO', 'LSTAT']) # this creates the design matrix for our feature vectors
Y = boston['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2) # split data into training and testing data.

est = sm.OLS(Y, X).fit()
print(est.summary())