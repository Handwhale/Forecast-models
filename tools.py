import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def RMSE(x, y):
    ((x - y) ** 2).mean() ** .5


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
        self.rmse_info.loc[-1] = [forecast_name, rmse]

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

    def SimpleExponentialSmoothing(self, alpha):
        # checked
        self.train_calc_data["SimpleExponentialSmoothing"] = pd.Series()

        # Set up 0 index
        self.train_calc_data["SimpleExponentialSmoothing"].iloc[0] = self.train_calc_data[self.value_column][0]

        # Training model
        for i in range(1, len(self.train_calc_data["SimpleExponentialSmoothing"])):
            self.train_calc_data["SimpleExponentialSmoothing"].iloc[i] = \
                self._SESEquation(alpha,
                                  self.train_calc_data[self.value_column][i - 1],
                                  self.train_calc_data["SimpleExponentialSmoothing"][i - 1])

        # Performing forecasting
        self.data_frame["SimpleExponentialSmoothing"] = \
            self._SESEquation(alpha,
                              self.train_calc_data[self.value_column][-1],
                              self.train_calc_data["SimpleExponentialSmoothing"][-1])

        # Output
        self._set_rmse("SimpleExponentialSmoothing")
        self._plot("SimpleExponentialSmoothing", "Simple Exponential Smoothing Forecast", self.rmse_info.RMSE[-1])


    def HoltModel(self, alpha, beta):
        # checked
        self.train_calc_data["HoltModel"] = pd.Series()

        self.train_calc_data["HoltModel"].iloc[0] = self.train_calc_data[self.value_column][0]

        for i in range(1, len(self.train_calc_data[self.value_column])):
            if i == 1:
                level = self.train_calc_data[self.value_column][0]
                trend = self.train_calc_data[self.value_column][1] - self.train_calc_data[self.value_column][0]

            prev_level = level
            level = alpha *  self.train_calc_data[self.value_column][i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            self.train_calc_data["HoltModel"].iloc[i] = level + trend

        self.data_frame["HoltModel"] = pd.Series()

        # Performing forecasting
        for i in range(len(self.test_data)):
            self.data_frame["HoltModel"].iloc[i] = level + trend * (i + 1)

        # Output
        self._set_rmse("HoltModel")
        self._plot("HoltModel", "Holt Model Forecast", self.rmse_info.RMSE[-1])


    def HoltWintersModel(self, alpha, beta, gamma, season_cycle):
        self.train_calc_data["HoltWintersSESEquation"] = pd.Series()
        self.train_calc_data["HoltWintersTrendEquation"] = pd.Series()
        self.train_calc_data["HoltWintersSeasonEquation"] = pd.Series()
        self.train_calc_data["HoltWintersModel"] = pd.Series()

        # Set up 0 index
        self.train_calc_data["HoltWintersSESEquation"].iloc[0] = self.train_calc_data[self.value_column][0]
        self.train_calc_data["HoltWintersTrendEquation"].iloc[0] = \
            self._TrendEquation(beta,
                                self.train_calc_data[self.value_column][0],
                                0,
                                self.train_calc_data[self.value_column][0])
        self.train_calc_data["HoltWintersSeasonEquation"].iloc[0] = \
            self._SeasonEquation(gamma,
                                 self.train_calc_data[self.value_column][0],
                                 self.train_calc_data[self.value_column][0],
                                 0)

        # Training model
        for i in range(1, len(self.train_calc_data["HoltWintersModel"])):
            current_season_index = season_cycle if i - season_cycle >= 0 else 0
            test_val = self.train_calc_data["HoltWintersSeasonEquation"][i - current_season_index]

            self.train_calc_data["HoltWintersSESEquation"].iloc[i] = \
                self._SESEquation(alpha,
                                  self.train_calc_data[self.value_column][i - 1],
                                  self.train_calc_data["HoltWintersSESEquation"][i - 1],
                                  self.train_calc_data["HoltWintersSeasonEquation"][i - 1 - current_season_index])

            self.train_calc_data["HoltWintersTrendEquation"].iloc[i] = \
                self._TrendEquation(beta,
                                    self.train_calc_data["HoltWintersSESEquation"][i],
                                    self.train_calc_data["HoltWintersTrendEquation"][i - 1],
                                    self.train_calc_data["HoltWintersSESEquation"][i - 1])

            self.train_calc_data["HoltWintersSeasonEquation"].iloc[i] = \
                self._SeasonEquation(gamma,
                                     self.train_calc_data[self.value_column][i - 1],
                                     self.train_calc_data["HoltWintersSESEquation"][i - 1],
                                     self.train_calc_data["HoltWintersSeasonEquation"][i - current_season_index])

            self.train_calc_data["HoltWintersModel"].iloc[i] = \
                self.train_calc_data["HoltWintersSESEquation"].iloc[i] + \
                self.train_calc_data["HoltWintersTrendEquation"].iloc[i] + \
                self.train_calc_data["HoltWintersSeasonEquation"][i + 1 - current_season_index]

        # plt.figure(figsize=(20, 10))
        # plt.plot(self.train_data.index, self.train_data[self.value_column], label='Train')
        # plt.plot(self.test_data.index, self.test_data[self.value_column], label='Test')
        # plt.plot(self.train_calc_data.index, self.train_calc_data["HoltWintersModel"], label="HoltWintersModel")
        # plt.legend(loc='best')
        # plt.title("LELE")
        # plt.show()

        print(self.train_calc_data["HoltWintersSESEquation"])

        # Performing forecasting
        # self.data_frame["HoltWintersModel"] = pd.Series()
        # for i in range(0, len(self.test_data[self.value_column])):
        #     self.data_frame["HoltWintersModel"].iloc[i] = \
        #         self.train_calc_data["HoltSESEquation"].iloc[-1] + \
        #         self.train_calc_data["HoltTrendEquation"].iloc[-1] * (i + 1) + \
        #         self.train_calc_data["HoltWintersSeasonEquation"].iloc[-1 ]

        # Output
        # self._set_rmse("HoltWintersModel")
        # self._plot("HoltWintersModel", "Holt-Winters Model Forecast", self.rmse_info.RMSE[-1])
