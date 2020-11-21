# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 23:08:12 2020

@author: Dipen Rupani
"""

import pandas as pd
import matplotlib.pyplot as plt

boston = pd.read_excel('BostonHousingData.xlsx') # loading in the data

x_1 = boston['RM'] # to run LR on var 1
x_2 = boston['LSTAT'] # to run LR on var 2
y = boston['MEDV'] # on MEDV


plt.figure(1)
plt.scatter(x_1, y, marker = 'o') # positive correlation capped at 50 with some outliers
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.plot()

plt.figure(2)
plt.scatter(x_2, y, marker = 'x') # negative correlation, not linear
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.plot()