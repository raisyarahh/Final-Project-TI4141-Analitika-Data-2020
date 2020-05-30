# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 00:14:26 2020

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_province = pd.read_csv('province_data.csv')
data_regional = pd.read_csv('regional_data.csv')

hospitalized = data_regional.groupby(['Date'])['Total_Hospitalized'].sum().reset_index()
new_postive = data_regional.groupby(['Date'])['New_Actually_Positive'].sum().reset_index()
healed = data_regional.groupby(['Date'])['Healed'].sum().reset_index()
death = data_regional.groupby(['Date'])['Deceased'].sum().reset_index()
Total = data_regional.groupby(['Date'])['Total_Cases'].sum().reset_index()

hos1 = hospitalized.T
new1 = new_postive.T
healed1 = healed.T
death1 = death.T
Total1 = Total.T

data_province.isnull().sum()
data_regional.isnull().sum()

import matplotlib.colors as mcolors
import random
import math
import time
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator
plt.style.use('seaborn')

from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

print(data_province.describe())
print(data_regional.describe())

print(data_province.groupby('Province_Code').size())
print(data_regional.groupby('Region_Code').size())

cols = data_regional.keys()
cols

dates = data_regional.keys()

days_since_1_60 = np.array([i for i in range(len(dates))]).reshape(-1,1)
Total1 = np.array(Total1).reshape(-1,1)
death1 = np.array(death1).reshape(-1,1)
healed1 = np.array(healed1).reshape(-1,1)

day_in_future = 10
future_forcast = np.array([i for i in range(len(dates)+day_in_future)]).reshape(-1,1)
adjusted_dates = future_forcast[:-10]

start = '2/24/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%y')
future_forecast_dates = []
for i in range(len(future_forcast)) :
    future_forecast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%y'))

latest_total = Total1[dates[-1]]
latest_deaths = death1[dates[-1]]
latest_healed = healed1[dates[-1]]