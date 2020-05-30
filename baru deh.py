# -*- coding: utf-8 -*-
"""
Created on Sun May 10 11:38:21 2020

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import random
import math
import time
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator
import seaborn as sns

#supervised learning python
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
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve

data_regional = pd.read_csv('regional_data.csv')
total9 = data_regional[(data_regional.Region_Code == 9)]
total9_total_cases = total9.groupby(['Date'])['Total_Cases'].sum().reset_index()
del total9_total_cases['Total_Cases']

totalcases9 = data_regional[(data_regional.Region_Code == 9)]
total9cases = totalcases9.groupby(['Date'])['Total_Cases'].sum().reset_index()
del total9cases['Date']

#ubah ke datetime
#merge
#bobo dulu