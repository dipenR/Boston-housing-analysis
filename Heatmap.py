# -*- coding: utf-8 -*-
"""
Created on Thu Mar 1 22:57:01 2020

@author: Dipen Rupani
"""

import pandas as pd
import seaborn as sns


boston = pd.read_excel('BostonHousingData.xlsx') # loading in the data

sns.set(rc = {'figure.figsize':(11.7, 8.27)}) # to format the heatmap 
sns.distplot(boston['MEDV'], bins = 30)

correlation_matrix = boston.corr().round(2) # first, we check which features have correlation with MEDV 
sns.heatmap(data = correlation_matrix, annot = True) # by plotting a heatmap