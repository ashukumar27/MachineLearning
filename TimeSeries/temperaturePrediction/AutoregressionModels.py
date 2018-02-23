#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:10:27 2018

@author: ashutosh

Time Series: AutoRegression Model
"""

"""
Autoregression is a time series model that uses observations from 
previous time steps as input to a regression equation to predict the 
value at the next time step.

Key Things:
* How to explore your time series data for autocorrelation.
* How to develop an autocorrelation model and use it to make predictions.
* How to use a developed autocorrelation model to make rolling predictions.
"""

"""
we can predict the value for the next time step (t+1) given the 
observations at the last two time steps (t-1 and t-2). 

As a regression model, this would look as follows:


X(t+1) = b0 + b1*X(t-1) + b2*X(t-2)
X(t+1) = b0 + b1*X(t-1) + b2*X(t-2)

Because the regression model uses data from the same input variable at 
previous time steps, it is referred to as an autoregression 
(regression of self).

"""

"""
Minimum Daily Temperatures Dataset
This dataset describes the minimum daily temperatures over
 10 years (1981-1990) in the city Melbourne, Australia.
"""


#Importing Data
import pandas as pd
import matplotlib.pyplot as plt
series = pd.read_csv('/Users/ashutosh/Documents/analytics/Projects/TimeSeries/daily-minimum-temperatures.csv', parse_dates = ['Date'])
print(series.head())
series['temp'].plot()
plt.show()

## Checking for Autocorrelation
from pandas.tools.plotting import lag_plot
lag_plot(series['temp'])
plt.show()

## Calculating Perason Coefficient between neighbouring 

values = pd.DataFrame(series['temp'].values)
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)

## Autocorrelation Plots

from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(series['temp'])
plt.show()


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series['temp'], lags=31)
plt.show()



#######################################################################
##  Baseline Model - Persistence Model
## Just select the previous days observation as next day's prediction

from sklearn.metrics import mean_squared_error

# create lagged dataset
values = pd.DataFrame(series['temp'].values)
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
# split into train and test sets
X = dataframe.values
train, test = X[1:len(X)-7], X[len(X)-7:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
 
# persistence model
def model_persistence(x):
	return x
 
# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
# plot predictions vs expected
plt.plot(test_y)
plt.plot(predictions, color='red')
plt.show()


#######################################################################
##  Autoregression Model
## Uses lagged variables as input variables



from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

# split dataset
X = series['temp'].values
train, test = X[1:len(X)-7], X[len(X)-7:]
# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


## Ading predicted observations to trainig
# split dataset
X = series['temp'].values
train, test = X[1:len(X)-7], X[len(X)-7:]
# train autoregression
model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


