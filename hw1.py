# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:03:10 2020

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

################################# ORDER #######################################
'''
    This codebase is meant to be run in the following order: 
        Heatmap.py
        PlottingSingle.py
        SimpleLinearRegression.py
        MultipleLinearRegression.py
        Residuals.py
        Summary.py
'''