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
        self.train_calc_data = train_data.copy()
        self.rmse_info = pd.DataFrame(columns=["Forecast Method", "RMSE"])

    def _set_rmse(self, forecast_name):
        rmse = ((self.data_frame[forecast_name] - self.test_data[self.value_column]) ** 2).mean() ** .5
        self.rmse_info.loc[-1] = [forecast_name, rmse]

        print("{} RMSE: {:.2f}".format(forecast_name, rmse))

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
        self.data_frame["SimpleAverage"] = self.train_data[self.value_column].mean()
        self._set_rmse("SimpleAverage")
        self._plot("SimpleAverage", "Simple Average Forecast", self.rmse_info.RMSE[-1])

    def MovingAverage(self, frame_size):
        self.data_frame["MovingAverage"] = self.train_data[self.value_column].rolling(frame_size).mean().iloc[-1]
        self._set_rmse("MovingAverage")
        self._plot("MovingAverage", "Moving Average Forecast", self.rmse_info.RMSE[-1])

    def SimpleExponentialSmoothing(self, a):

        self.train_calc_data["SimpleExponentialSmoothing"] = pd.Series()

        # Calc known values
        self.train_calc_data["SimpleExponentialSmoothing"].iloc[0] = self.train_calc_data[self.value_column][0]
        for i in range(1, len(self.train_calc_data["SimpleExponentialSmoothing"])):
            self.train_calc_data["SimpleExponentialSmoothing"].iloc[i] = \
                self._SES(a, self.train_calc_data[self.value_column][i - 1],
                          self.train_calc_data["SimpleExponentialSmoothing"][i - 1])

        # Performing forecasting
        self.data_frame["SimpleExponentialSmoothing"] = \
            self._SES(a, self.train_calc_data[self.value_column][-1],
                      self.train_calc_data["SimpleExponentialSmoothing"][-1])

        # print(self.data_frame["SimpleExponentialSmoothing"])

        self._set_rmse("SimpleExponentialSmoothing")
        self._plot("SimpleExponentialSmoothing", "Simple Exponential Smoothing Forecast", self.rmse_info.RMSE[-1])

    def _SES(self, alpha, actual, prev_forecast):
        return alpha * actual + (1 - alpha) * prev_forecast

    def HoltModel(self):
        pass

    def HoltWintersModel(self):
        pass

