import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def RMSE(x, y):
    ((x - y) ** 2).mean() ** .5


class ForecastModels:
    def __init__(self, train_data, test_data, value_column):
        self.train_data = train_data
        self.test_data = test_data
        self.value_column = value_column
        self.data_frame = test_data.copy()
        self.rmse_info = pd.DataFrame(columns=["Forecast Method", "RMSE"])

    def _set_rmse(self, forecast_name):
        rmse = ((self.data_frame[forecast_name] - self.test_data[self.value_column]) ** 2).mean() ** .5
        self.rmse_info.loc[-1] = [forecast_name, rmse]

        print("{} RMSE: {}".format(forecast_name, rmse))

    def _plot(self, column_name, label, rmse_value):
        plt.figure(figsize=(20, 10))
        plt.plot(self.train_data.index, self.train_data[self.value_column], label='Train')
        plt.plot(self.test_data.index, self.test_data[self.value_column], label='Test')
        plt.plot(self.data_frame.index, self.data_frame[column_name], label=column_name)
        plt.figtext(0.13, .78, "RMSE = {0:.2f}".format(rmse_value))
        plt.legend(loc='best')
        plt.title(label)
        plt.show()

    def NaiveForecast(self):
        self.data_frame["Naive"] = self.train_data[self.value_column][len(self.train_data[self.value_column]) - 1]
        self._set_rmse("Naive")
        self._plot("Naive", "Naive Forecast", self.rmse_info.RMSE[-1])

    def SimpleAverage(self):
        pass

    def MovingAverage(self):
        pass

    def SimpleExponentialSmoothing(self):
        pass

    def HoltModel(self):
        pass

    def HoltWintersModel(self):
        pass


