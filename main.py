import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tools

# Get data
df = pd.read_csv('Train.csv')
# df.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)
# plt.show()


# Split data
train = df[0:12800]
test = df[12800:]
# train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)
# test.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)
# plt.show()

# Aggregating the dataset at daily level
df["Timestamp"] = pd.to_datetime(df["Datetime"], format='%d-%m-%Y %H:%M')
df.index = df.Timestamp
df = df.resample('D').mean()
train["Timestamp"] = pd.to_datetime(train["Datetime"], format='%d-%m-%Y %H:%M')
train.index = train.Timestamp
train = train.resample('D').mean()
test["Timestamp"] = pd.to_datetime(test["Datetime"], format='%d-%m-%Y %H:%M')
test.index = test.Timestamp
test = test.resample('D').mean()

# Data decomposition
# import statsmodels.api as sm
# sm.tsa.seasonal_decompose(train.Count).plot()
# result = sm.tsa.stattools.adfuller(train.Count)
# plt.show()


models = tools.ForecastModels(train, test, "Count")
# models.NaiveForecast()
# models.SimpleAverage()
# models.MovingAverage(60)
# models.SimpleExponentialSmoothing(0.1)
models.HoltModel(.1, .3)
