import pandas as pd
import numpy as np
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
        # print("{} RMSE: {:.2f}".format(forecast_name, rmse))
        return rmse

    def _plot(self, column_name, label, rmse_value):
        plt.figure(figsize=(15, 8))
        plt.plot(self.train_data.index, self.train_data, label='Train')
        # plt.plot(self.test_data.index, self.test_data, label='Test')
        plt.plot(self.data_frame.index, self.data_frame[column_name], label=column_name)
        # plt.figtext(0.13, .78, "RMSE = {0:.2f}".format(rmse_value))
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

    def SimpleExponentialSmoothing(self, alpha, grid_search=False):
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
        if not grid_search:
            val = self._set_rmse("SimpleExponentialSmoothing")
            self._plot("SimpleExponentialSmoothing", "Simple Exponential Smoothing Forecast", val)
        else:
            return self._set_rmse("SimpleExponentialSmoothing")

    def HoltModel(self, alpha, beta, grid_search=False):
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
        if not grid_search:
            val = self._set_rmse("HoltModel")
            self._plot("HoltModel", "Holt Model Forecast", val)
        else:
            return self._set_rmse("HoltModel")

    def HoltWinter(self, alpha, beta, gamma, season_cycle, grid_search=False):
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
        if not grid_search:
            val = self._set_rmse("WinterHolt")
            self._plot("WinterHolt", "Winter Holt Model Forecast", val)
        else:
            return self._set_rmse("WinterHolt")

    def RMSEResults(self):
        print(self.rmse_info)


class GridSearch:
    def __init__(self, fc_models):

        self.fc_models = fc_models

    def SimpleExp(self):
        max_param = 0
        max_result = 100000

        for a in np.arange(0.0, 1.1, 0.01):
            result = self.fc_models.SimpleExponentialSmoothing(a, grid_search=True)
            if result < max_result:
                max_result = result
                max_param = a

        print("Simple Exp optimal param = {} with the result {}".format(max_param, max_result))
        return max_param

    def DoubleExp(self):
        max_param_a = 0
        max_param_b = 0
        max_result = 100000

        for a in np.arange(0.0, 1.1, 0.01):
            for b in np.arange(0.0, 1.1, 0.01):
                result = self.fc_models.HoltModel(a, b, grid_search=True)
                if result < max_result:
                    max_result = result
                    max_param_a = a
                    max_param_b = b

        print("Simple Exp optimal param = {} and {} with the result {}".format(max_param_a, max_param_b, max_result))

    def TripleExp(self):
        max_param_a = 0
        max_param_b = 0
        max_param_c = 0
        max_result = float("inf")

        simple_range = np.arange(0.0, 1.1, 0.1)

        for a in simple_range:
            for b in simple_range:
                for c in simple_range:
                    result = self.fc_models.HoltWinter(a, b, c, 12 * 4, grid_search=True)
                    if (result) < max_result:
                        max_result = result
                        max_param_a = a
                        max_param_b = b
                        max_param_c = c

        print("1st iteration grid search: {} and {} and {} with the result {}".format(max_param_a, max_param_b,
                                                                                      max_param_c, max_result))

        step = 0.01

        a_start_range = (max_param_a - 0.1) if max_param_a - 0.1 >= 0 else 0
        b_start_range = (max_param_b - 0.1) if max_param_b - 0.1 >= 0 else 0
        c_start_range = (max_param_c - 0.1) if max_param_c - 0.1 >= 0 else 0

        a_end_range = (max_param_a + 0.1 + step) if max_param_a + 0.1 <= 1 else 0
        b_end_range = (max_param_b + 0.1 + step) if max_param_b + 0.1 <= 1 else 0
        c_end_range = (max_param_c + 0.1 + step) if max_param_c + 0.1 <= 1 else 0

        a_range = np.arange(a_start_range, a_end_range, step)
        b_range = np.arange(b_start_range, b_end_range, step)
        c_range = np.arange(c_start_range, c_end_range, step)

        for a in a_range:
            for b in b_range:
                for c in c_range:
                    result = self.fc_models.HoltWinter(a, b, c, 12 * 4, grid_search=True)
                    print("{} nd {} nd {} result {}".format(a, b, c, result))
                    if (result) < max_result:
                        max_result = result
                        max_param_a = a
                        max_param_b = b
                        max_param_c = c

        print("2nd iteration grid search: {} and {} and {} with the result {}".format(max_param_a, max_param_b,
                                                                                      max_param_c, max_result))
