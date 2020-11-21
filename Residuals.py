# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 23:23:45 2020

@author: Dipen Rupani
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot

boston = pd.read_excel('BostonHousingData.xlsx') # loading in the data

X = pd.DataFrame(np.c_[boston['INDUS'], boston['NOX'], boston['RM'], boston['TAX'], boston['PTRATIO'],boston['LSTAT']], columns = ['INDUS', 'NOX', 'RM', 'TAX', 'PTRATIO', 'LSTAT']) # this creates the design matrix for our feature vectors
Y = boston['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2) # split data into training and testing data.

lin_model = LinearRegression() # creates a linear regression object with coeffeicients to minimize the residual sum of squares between the observed targets and and predicted targets.
visualizer = ResidualsPlot(lin_model)
visualizer.fit(X_train, Y_train) # fits data into the linear model - training the model
visualizer.score(X_test, Y_test)
visualizer.show()