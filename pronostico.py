# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:19:19 2020

@author: jasso
"""

import pandas as pd
import numpy as np
import collections
import datetime 

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

from matplotlib import pyplot
#from statsmodels.graphics.tsaplots import plot_acf
#from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
#from sklearn.metrics import mean_squared_error
# from statsmodels.tsa.stattools import adfuller
import warnings
# import numpy
from sklearn.metrics import mean_squared_error
#import datetime
from statistics import mode


series = pd.read_csv('ExporteEmpresas.csv', sep=';', header=0, index_col=None)
serie = series.loc[:,['Fecha','Inicio Intervalo','Nombre Skill','LLamadas Ofrecidas','Llamadas Atendidas']]
serie = serie.assign(DateTime = serie['Fecha'] + ' ' + serie['Inicio Intervalo'] )
del(serie['Fecha'],serie['Inicio Intervalo'])
serie = serie.iloc[:,[3,0,1,2]] 
skills = serie['Nombre Skill'].drop_duplicates()
dtime = pd.DataFrame(serie['DateTime'].drop_duplicates())
dtime = dtime.assign(LLamadasOfrecidas=np.zeros((dtime.shape[0],1)),
                     LlamadasAtendidas=np.zeros((dtime.shape[0],1))).to_numpy()
serie = serie.to_numpy()

min_hour = 60
hour_day = 24
interval = 30

max_blocks = (min_hour*hour_day)/interval

dic_skills = {}
for x in skills:
    row = []
    for y in range(serie.shape[0]): 
        if serie[y,1] == x:
            row.append(list([serie[y,0],serie[y,2],serie[y,3]])) 
    row = np.array(row)        
    dic_skills[x] = row

for x in dic_skills:
    for y in dic_skills[x]:
        pos = np.where(dtime[:,0] == y[0])
        dtime[pos,1] += int(y[1])
        dtime[pos,2] += int(y[2])
        
dtime = pd.DataFrame(dtime)        
dtime['Fecha'], dtime['time'] = dtime[0].str.split(" ", 1).str
del(dtime[0])
dtime = dtime.iloc[:,[2,3,0,1]] 

date = dtime['Fecha'].drop_duplicates()
date = date.to_numpy()
dtime = dtime.to_numpy()

dic_date = collections.OrderedDict()
for x in date:
    pos = np.where(dtime[:,0] == x)
    time = []
    for y in pos:
        time.append([dtime[y,1],dtime[y,2],dtime[y,3]])
    time = np.array(time)
    dic_date[x] = time 
    
Info_ofrecidas = []
Info_atendidas = [] 
for y in dic_date:
    init = datetime.datetime.strptime(dic_date[y][0][0][0], '%H:%M').time()
    stop = datetime.datetime.strptime(dic_date[y][0][0][-1], '%H:%M').time()        
    init_block = (init.hour*min_hour + init.minute)/interval
    stop_block = (stop.hour*min_hour + stop.minute)/interval
    # init_zeros_values = []
    # stop_zeros_values = []
    init_count = 1
    while init_count < int(init_block):
      #h,h1 = divmod(init_count * interval,60)              
      #init_values = []
      #init_zeros_values.append([datetime.time(int(h),int(h1)),0,0])
      Info_ofrecidas += [0]
      Info_atendidas += [0]
      #init_zeros_values.append(init_values)
      init_count += 1
    stop_count = stop_block
    Info_ofrecidas += [x for x in dic_date[y][0][1]]
    Info_atendidas += [x for x in dic_date[y][0][2]]
    stop_count = stop_block
    while stop_count < int((hour_day*min_hour)/interval):
      #h,h1 = divmod(init_count * interval,60)
      #stop_zeros_values.append([datetime.time(int(h),int(h1)),0,0])
      Info_ofrecidas += [0]
      Info_atendidas += [0]
      stop_count += 1 



# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

days_in_year = 96*7*4
differenced = difference(Info_atendidas, days_in_year)
# fit model
model = ARIMA(differenced, order=(8,0,0))
model_fit = model.fit(disp=-1)
forecast = model_fit.forecast(steps=96*7*4*3)[0]

def forescast_model(history):
    predict = list()
    day = 1
    for yhat in forecast:
        inverted = inverse_difference(history, yhat, days_in_year)
        print('interval %d: %f' % (day, inverted))
        history.append(inverted)
        predict.append(inverted)
        day += 1
    return predict    

history = [x for x in Info_atendidas]

for x in range(3):
    predict = forescast_model(history)
    day = 1
    for y in predict:
        history.append(y-mode(predict))
                       #mode(predict))
        



# # create a differenced series
# def difference(dataset, interval=1):
# 	diff = list()
# 	for i in range(interval, len(dataset)):
# 		value = dataset[i] - dataset[i - interval]
# 		diff.append(value)
# 	return np.array(diff)

# # invert differenced value
# def inverse_difference(history, yhat, interval=1):
# 	return yhat + history[-interval]

# for x in range(3):
#     days_in_year = 96*7*4
#     differenced = difference(history2, days_in_year)
#     # fit model
#     model = ARIMA(differenced, order=(8,0,0))
#     model_fit = model.fit(disp=-1)
#     # multi-step out-of-sample forecast
#     forecast = model_fit.forecast(steps=96*7*4)[0]
#     # invert the differenced forecast to something usable
#     history = [x for x in history2]
#     forcast = list()
#     day = 1
#     for yhat in forecast:
#         inverted = inverse_difference(history, yhat, days_in_year)
#         print('interval %d: %f' % (day, inverted))
#         history.append(inverted)
#         forcast.append(inverted)
#         day += 1
