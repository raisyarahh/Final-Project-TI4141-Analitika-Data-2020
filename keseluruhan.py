# -*- coding: utf-8 -*-
"""
Created on Sun May 10 17:16:02 2020

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

data_province = pd.read_csv('province_data.csv')
data_regional = pd.read_csv('regional_data.csv')
#data eksternal, data indonesia
data_indonesia = pd.read_csv('Kasus Harian Indonesia.csv')
tc1 = data_regional.groupby(['Date'])['Total_Cases', 'New_Actually_Positive'].sum().reset_index()
tc1['New_Actually_Positive'].mean()

tc1.to_excel(r'D:\sarah\Akademik\smt 6 TI\andat\uas\tc1.xlsx', index = False)

total_cases = data_regional.groupby(['Date'])['Total_Cases'].sum().reset_index()
total_death = data_regional.groupby(['Date'])['Deceased'].sum().reset_index()
total_healed = data_regional.groupby(['Date'])['Healed'].sum().reset_index()

FMT = '%Y-%m-%d %H:%M:%S'
date = total_cases['Date']
total_cases['Date'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime('2020-02-24 18:00:00' , FMT)).days)

def logistic_model(x,a,b,c) :
    return c/(1+np.exp(-(x-b)/a))

x = list(total_cases.iloc[1:,0])
y = list(total_cases.iloc[1:,1])
fit = curve_fit(logistic_model,x,y)

A,B = fit
A

errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]
errors

a=A[0]+errors[0]
b=A[1]+errors[1]
c=A[2]+errors[2]

#jumlah hari maksimal sejak 24 februari
sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))
sol

pred_x = list(range(max(x),sol))
plt.rcParams['figure.figsize']=[7,7]
plt.rc('font', size=14)
plt.scatter(x,y,label='Real data', color='red')

#logistic curve
plt.plot(x+pred_x, [logistic_model(i,a,b,c) for i in x+pred_x], label='Logistic model')

plt.legend()
plt.xlabel('Days since 24 feb 2020')
plt.ylabel('Total cases')
plt.ylim((min(y)*0.9,c*1.1))
plt.show()

y_pred_logistic = [logistic_model(i,a,b,c) for i in x]
p=mean_squared_error(y,y_pred_logistic)

s1=(np.subtract(y,y_pred_logistic)**2).sum()
s2=(np.subtract(y,np.mean(y))**2).sum()
r=1-s1/s2
print('R^2 {}'.format(r))
print('Mean square error {}'.format(p))

print('Jumlah kasus maksimal menurut prediksi adalah {:f}'.format(A[2]+errors[2])) #Penambahan dengan error
print('Puncak wabah adalah {:.0f} hari setelah 24 feb 2020 atau {}'. format(sol,x))

total_kasus = data_regional.groupby(['Date'])['Total_Cases','Healed','Deceased'].sum().reset_index()
plt.plot(total_kasus['Total_Cases'], color='blue')
plt.plot(total_kasus['Healed'], color='green')
plt.plot(total_kasus['Deceased'], color='red');

#indonesia
plt.plot(data_indonesia['Kasus (Kumulatif)'], color='blue')
plt.plot(data_indonesia['Sembuh Kumulatif'], color='green')
plt.plot(data_indonesia['Meninggal Kumulatif'], color='red');