import pandas as pd

import matplotlib.pyplot as plt

class ForecastModels:
    def __init__(self, train_data, test_data, value_column):
        self.value_column = value_column
        self.data_frame = test_data.copy()
        self.train_calc_data = train_data.copy()

        self.train_data = self.train_calc_data[self.value_column]
        self.test_data = self.data_frame[self.value_column]

        self.rmse_info = pd.DataFrame(columns=["Forecast Method", "RMSE"])

    def _set_rmse(self, forecast_name):
        rmse = ((self.data_frame[forecast_name] - self.test_data) ** 2).mean() ** .5
        self.rmse_info.loc[len(self.rmse_info)] = [forecast_name, rmse]
        return rmse

        print("{} RMSE: {:.2f}".format(forecast_name, rmse))

    def _plot(self, column_name, label, rmse_value):
        plt.figure(figsize=(20, 10))
        plt.plot(self.train_data.index, self.train_data, label='Train')
        plt.plot(self.test_data.index, self.test_data, label='Test')
        plt.plot(self.data_frame.index, self.data_frame[column_name], label=column_name)
        plt.figtext(0.13, .78, "RMSE = {0:.2f}".format(rmse_value))
        plt.legend(loc='best')
        plt.title(label)
        plt.show()

    def _SESEquation(self, alpha, actual, prev_forecast, prev_trend_forecast=0, prev_season_forecast=0):
        return alpha * (actual - prev_season_forecast) + (1 - alpha) * (prev_forecast - prev_trend_forecast)

    def _TrendEquation(self, beta, ses_result, prev_forecast, prev_ses_forecast):
        return beta * (ses_result - prev_ses_forecast) + (1 - beta) * prev_forecast

    def _SeasonEquation(self, gamma, actual, ses_result, prev_forecast):
        return gamma * (actual - ses_result) + (1 - gamma) * prev_forecast

    def NaiveForecast(self):
        self.data_frame["Naive"] = self.train_data[len(self.train_data) - 1]
        val = self._set_rmse("Naive")
        self._plot("Naive", "Naive Forecast", val)

    def SimpleAverage(self):
        self.data_frame["SimpleAverage"] = self.train_data.mean()
        val = self._set_rmse("SimpleAverage")
        self._plot("SimpleAverage", "Simple Average Forecast", val)

    def MovingAverage(self, frame_size):
        self.data_frame["MovingAverage"] = self.train_data.rolling(frame_size).mean().iloc[-1]
        val = self._set_rmse("MovingAverage")
        self._plot("MovingAverage", "Moving Average Forecast", val)

    def SimpleExponentialSmoothing(self, alpha):
        # checked
        self.train_calc_data["SimpleExponentialSmoothing"] = pd.Series()

        # Set up 0 index
        self.train_calc_data["SimpleExponentialSmoothing"].iloc[0] = self.train_data[0]

        # Training model
        for i in range(1, len(self.train_calc_data["SimpleExponentialSmoothing"])):
            self.train_calc_data["SimpleExponentialSmoothing"].iloc[i] = \
                self._SESEquation(alpha,
                                  self.train_data[i - 1],
                                  self.train_calc_data["SimpleExponentialSmoothing"][i - 1])

        # Performing forecasting
        self.data_frame["SimpleExponentialSmoothing"] = \
            self._SESEquation(alpha,
                              self.train_data[-1],
                              self.train_calc_data["SimpleExponentialSmoothing"][-1])

        # Output
        val = self._set_rmse("SimpleExponentialSmoothing")
        self._plot("SimpleExponentialSmoothing", "Simple Exponential Smoothing Forecast", val)

    def HoltModel(self, alpha, beta):
        # checked
        self.train_calc_data["HoltModel"] = pd.Series()

        self.train_calc_data["HoltModel"].iloc[0] = self.train_data[0]

        for i in range(1, len(self.train_data)):
            if i == 1:
                level = self.train_data[0]
                trend = self.train_data[1] - self.train_data[0]

            prev_level = level
            level = alpha * self.train_data[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            self.train_calc_data["HoltModel"].iloc[i] = level + trend

        self.data_frame["HoltModel"] = pd.Series()

        # Performing forecasting
        for i in range(len(self.test_data)):
            self.data_frame["HoltModel"].iloc[i] = level + trend * (i + 1)

        # Output
        val = self._set_rmse("HoltModel")
        self._plot("HoltModel", "Holt Model Forecast", val)

    def HoltWinter(self, alpha, beta, gamma, season_cycle):
        self.train_calc_data["WinterHolt"] = pd.Series()
        # Set up 0 index
        level = self.train_data[0]
        trend = self.train_data[1] - self.train_data[0]
        seasonals = []
        for j in range(season_cycle):
            seasonals.append(self.train_data[j + season_cycle] - self.train_data[j])

        for i in range(len(self.train_data)):
            prev_level = level
            level = alpha * (self.train_data[i] - seasonals[i % season_cycle]) + (1 - alpha) * (prev_level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            seasonals[i % season_cycle] = gamma * (self.train_data[i] - level) + \
                                          (1 - gamma) * seasonals[i % season_cycle]
            self.train_calc_data["WinterHolt"].iloc[i] = level + trend + seasonals[i % season_cycle]

        self.data_frame["WinterHolt"] = pd.Series()
        # Performing forecasting
        for i in range(len(self.test_data)):
            self.data_frame["WinterHolt"].iloc[i] = level + trend * (i + 1) + seasonals[i % season_cycle]

        # Output
        val = self._set_rmse("WinterHolt")
        self._plot("WinterHolt", "Winter Holt Model Forecast", val)

    def RMSEResults(self):
        print(self.rmse_info)
