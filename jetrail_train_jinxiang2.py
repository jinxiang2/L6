#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
from fbprophet import Prophet
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

# 读入数据集
train = pd.read_csv('./train.csv')
print(train.head())
print(train.tail())

train['Datetime'] = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
train.index = train.Datetime
print(train.head())

train.drop(['ID','Datetime'], axis=1, inplace=True)
print(train.head())
daily_train = train.resample('D').sum()
daily_train['ds'] = daily_train.index
daily_train['y'] = daily_train.Count
daily_train.drop(['Count'], axis=1, inplace=True)
print(daily_train.head())

m = Prophet(yearly_seasonality= True, seasonality_prior_scale=0.1)
m.fit(daily_train)
future = m.make_future_dataframe(periods= 213)
forcast = m.predict(future)
print(forcast)

m.plot(forcast)

m.plot_components(forcast)


