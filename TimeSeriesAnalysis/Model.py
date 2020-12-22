#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2020 by leodflag

"""This module contains time series models for forecasting.

Time series prediction models include autoregressive models and moving average models, which can be used for 
model training and model prediction, and model parameters such as prediction results and prediction errors can be 
obtained through the created objects.

Class 1
---------
AutoRegressive
	Autoregressive is a time series model.

Methods
---------
fit(train_data: pandas.Series)
    Fit autoregression model.

predict(test_data: pandas.Series) -> list
    Use an autoregressive model to make predictions.

Class 2
---------
SimpleMovingAverage
	Simple moving average is often used in the financial field..

Methods
---------
fit(train_data: pandas.Series)
    Fit simple moving average.

Notes
-----------
The input data format must be'pandas.Series'.

If the time series data type is a list, you can use the function "list_to_dataframe" of the module "ProcessData" to convert the data type.

It is recommended to use the function of'ProcessData' to obtain and process time series data, which is less prone to errors when entering the model.

"""
import pandas
import numpy
import TimeSeriesAnalysis.ProcessData as Data
import Math.Statistics as Math

class AutoRegressive:

    """Autoregressive is a time series model.

    Autoregressive is a time series model. The model uses past data in the time series as its own explanatory variable, so it will be affected by current errors and past data.

    Parameters
    ---------
        lags: int, default = 1
        Number of lagging periods. Decide to use the data of the previous periods as the explanatory variable. 
        The default value is 1, which means that only the data from the previous period is used as the explanatory variable, which is called the first order autoregressive model (AR(1)).

    Attributes
    ---------
        __data: pandas.Series
        One-dimensional time series data.
        For example:
            date
            2020-06-23 01:00:00    28.89
            2020-06-24 01:00:00    28.91
            2020-06-29 01:00:00    28.87
            2020-06-30 01:00:00    28.92
            2020-07-01 01:00:00    28.79
            Name: close, Length: 170, dtype: float64

        __ar_lags_data: pandas.DataFrame
        Autoregressive data set.
        For example AR(1) data set:
            date　　　　　　　　t-1　　t
            2020-09-29 23:00:00  29.10  28.89
            2020-09-30 23:00:00  28.89  28.91
            2020-10-01 23:00:00  28.91  28.87
            2020-10-04 23:00:00  28.87  28.92
            2020-10-06 13:56:21  28.92  28.79
            -------------------------------------------
            't': Current time series data
            't-1': Previous time series data

        __ar_test_data: pandas.DataFrame
        Autoregressive data set when using 'predict' function.

        __ar_coef_matrix: numpy.ndarray
        Multidimensional linear regression matrix normal equation.

        __ar_train_predict: float
        The error between the current period and the previous period.

        __ar_train_error: list
        Time series forecast results of autoregressive model. The order is [mse, mae, rmse, nrmse].

        __ar_test_data: list
        Time series forecast results of autoregressive model.

        __ar_test_predict: list
        Time series forecast results of autoregressive model.

        __ar_test_error: float
        The mean square error predicted by the autoregressive model. The order is [mse, mae, rmse, nrmse].

    Methods
    ---------
        lags -> int
            Return the number of lagging periods

        ar_lags_data -> pandas.DataFrame:
        Return autoregressive model data of lagging periods.

        test_data -> pandas.Series
            Return data used for testing.

        coefficients_matrix -> numpy.ndarray
            Return the autoregressive model coefficients matrix.

        train_predict -> pandas.Series
            Return the results predicted by the autoregressive model using the training set.

        test_predict -> pandas.Series
            Return the results predicted by the autoregressive model using the testing set.

        train_error -> list
            Return the regression index used to calculate the training prediction error. The order is [mse, mae, rmse, nrmse.

        test_error -> list
            Return the regression index used to calculate the testing prediction error. The order is [mse, mae, rmse, nrmse].

        fit(train_data: pandas.Series)
            Fit autoregression model.

        predict(test_data: pandas.Series) -> list
            Use an autoregressive model to make predictions.

    Examples
    ---------
        import TimeSeriesAnalysis.ProcessData as Data
        import TimeSeriesAnalysis.Model as model

        col_name = 'close'
        file_path = 'test_data/TWD_unittest.csv'
        times_data_id = 'TWD'
        save_path = 'TWD'

        # TimeSeriesAnalysis.ProcessData
        data = Data.read_file(file_path, col_name)
        train, test = Data.split_data(data, 0.7)

        # TimeSeriesAnalysis.Model
        model = model.AutoRegressive(lags = 2)
        model.fit(train)
        model.predict(test, pure_test_set_predict = True)
        print(model.test_error)  # [0.07378008693836643, 0.23415224307538107, 0.2716249011750698, 1.0278000869996953]

    References
    ---------
    Autoregressive model: https://zh.wikipedia.org/wiki/%E8%87%AA%E8%BF%B4%E6%AD%B8%E6%A8%A1%E5%9E%8B
    Least Squares Estimation of the Parameters: https://slideplayer.com/slide/4238175/14/images/12/12-1+Multiple+Linear+Regression+Models.jpg
    Matrix Approach to Multiple Linear Regression: https://slideplayer.com/slide/4238175/14/images/20/12-1+Multiple+Linear+Regression+Models.jpg

    """

    def __init__(self, lags: int = 1):
        self.__data = []
        self.__lags = lags
        self.__ar_lags_data = []
        self.__ar_coef_matrix = []
        self.__ar_train_predict = []
        self.__ar_train_error = [0,0,0,0] # mse  mae rmse nrmse
        self.__ar_test_data = []
        self.__ar_test_predict = []
        self.__ar_test_error = [0,0,0,0]

    @property
    def lags(self) -> int:
        """Return the number of lagging periods."""
        return self.__lags

    @property
    def ar_lags_data(self) -> pandas.DataFrame:
        """Return autoregressive model data of lagging periods."""
        return self.__ar_lags_data

    @property
    def test_data(self) -> pandas.Series:
        """Return data used for testing."""
        return self.__ar_test_data

    @property
    def coefficients_matrix(self) -> numpy.ndarray:
        """Return the autoregressive model coefficients matrix. """
        return self.__ar_coef_matrix

    @property
    def train_predict(self) -> pandas.Series:
        """Return the results predicted by the autoregressive model using the training set."""
        return self.__ar_train_predict['t']

    @property
    def test_predict(self) -> pandas.Series:
        """Return the results predicted by the autoregressive model using the testing set."""
        return self.__ar_test_predict['t']

    @property
    def train_error(self) -> list:
        """Return the regression index used to calculate the training prediction error. The order is [mse, mae, rmse, nrmse]."""
        return self.__ar_train_error

    @property
    def test_error(self) -> list:
        """Return the regression index used to calculate the testing prediction error. The order is [mse, mae, rmse, nrmse]."""
        return self.__ar_test_error

    def __cal_ar_normal_equation(self):
        """Multidimensional linear regression matrix normal equation."""
        leng = self.__lags + 1
        A_matrix = numpy.zeros([leng, leng])
        B_matrix = numpy.zeros([leng, 1])
        for i in range(leng):
            for j in range(leng):
                if (i == 0 and j == 0):
                    A_matrix[i][j] = len(self.__ar_lags_data['t'])
                    B_matrix[i][0] = self.__ar_lags_data['t'].sum()
                elif (i == 0 and i != j):
                    A_matrix[i][j] = self.__ar_lags_data['t-' + str(j)].sum()
                    B_matrix[j][0] = (self.__ar_lags_data['t']*self.__ar_lags_data['t-' + str(j)]).sum()
                elif (j == 0 and j !=i ):
                    A_matrix[i][j] = self.__ar_lags_data['t-' + str(i)].sum()
                else:
                    A_matrix[i][j] = (self.__ar_lags_data['t-'+str(i)] * self.__ar_lags_data['t-'+str(j)]).sum()
        A_inv = numpy.linalg.inv(A_matrix)
        coef_matrix = A_inv.dot(B_matrix)
        return coef_matrix

    def __predict_train(self):
        """Use an autoregressive model to predict the training set data."""
        predict = []
        for day in range(len(self.__ar_lags_data['t'])):
            pre = self.__ar_coef_matrix[0][0]
            for i in range(1, len(self.__ar_coef_matrix)):
                pre += self.__ar_coef_matrix[i][0]*self.__ar_lags_data['t-'+str(i)][day]
            predict.append(pre)
        self.__ar_train_predict = pandas.DataFrame({ 't': predict}, index = self.__ar_lags_data['t'].index)
        self.__ar_train_error[0] = Math.mean_square_error(self.__ar_lags_data['t'], self.__ar_train_predict['t'])
        self.__ar_train_error[1] = Math.mean_absolute_error(self.__ar_lags_data['t'], self.__ar_train_predict['t'])
        self.__ar_train_error[2] = Math.root_mean_squard_error(self.__ar_lags_data['t'], self.__ar_train_predict['t'])
        self.__ar_train_error[3] = Math.normalized_mean_squard_error(self.__ar_lags_data['t'], self.__ar_train_predict['t'])

    def fit(self, train_data: pandas.Series):
        """Fit autoregression model.

        Parameters
        ---------
            train_data: pandas.Series.
            Data for training the autoregression model.'train_data' is a one-dimensional numerical data of type'pandas.Series' taken from'pandas.DataFrame'.

        Returns
        ---------
        No return value.

        Error
        ---------
            ValueError: The type of one-dimensional data must be 'pandas.Series', and the content is numeric data.
            Solution: Check whether the parameter matches the 'ValueError' description, or use the default value.

            ValueError: The list must be one-dimensional numerical data, and there is at least one numerical data in it.
            Solution: The type of one-dimensional data must be 'pandas.Series', and the content is numeric data.

            ValueError: 'lags' value must be a positive integer, and the condition is greater than 1 and less than the length of the data.
            Solution: Check whether the parameter matches the 'ValueError' description, or use the default value.

            TypeError: object of type 'NoneType' has no len().
            Solution:  Please do not enter 'None'. The type of one-dimensional data must be 'pandas.Series', and the content is numeric data.

            KeyError: The 'train_data' type is'pandas.DataFrame', please enter the correct key value.
            Solution: Use the correct key value to obtain data.

        """
        self.__data = train_data
        self.__ar_lags_data = Data.create_ar_data(self.__data, self.__lags)
        self.__ar_coef_matrix = self.__cal_ar_normal_equation()
        self.__predict_train()

    def predict(self, test_data: pandas.Series, pure_test_set_predict: bool = True) -> pandas.Series:
        """Use an autoregressive model to make predictions.

        Parameters
        ---------
            test_data: pandas.Series.
            Data for testing the autoregression model.'test_data' is a one-dimensional numerical data of type'pandas.Series' taken from'pandas.DataFrame'.

            pure_test_set_predict: bool, default = True.
            The autoregressive model consumes data with a length of 'lags' when predicting.
            'pure_test_set_predict' default is true, which means that only the data of the test set is used for prediction, so the predicted result will be less than the number of'lags' data. 
            If the setting is false, the end data of the training set will be used as the starting point of the test set prediction, so the predicted data length will be the same as the test set data length.
            If 'pure_test_set_predict' is false, the end data of the training set will be used as the starting point of the test set prediction, so the predicted data length will be the same as the test set data length.

        Returns
        ---------
        No return value.

        Error
        ---------
            ValueError: The type of one-dimensional data must be 'pandas.Series', and the content is numeric data.
            Solution: Check whether the parameter matches the 'ValueError' description, or use the default value.

            ValueError: The list must be one-dimensional numerical data, and there is at least one numerical data in it.
            Solution: The type of one-dimensional data must be 'pandas.Series', and the content is numeric data.

            TypeError: object of type 'NoneType' has no len().
            Solution:  Please do not enter 'None'. The type of one-dimensional data must be 'pandas.Series', and the content is numeric data.

            KeyError: The 'test_data' type is'pandas.DataFrame', please enter the correct key value.
            Solution: Use the correct key value to obtain data.

        """

        if(pure_test_set_predict == True):
            predict = Data.dataframe_to_list(test_data[:self.__lags])
            for day in range(len(test_data[self.__lags:])):
                pre =  self.__ar_coef_matrix[0][0]
                for i in range(self.__lags, 0, -1):
                    pre += self.__ar_coef_matrix[i][0]*predict[day]
                    day += 1
                day -= self.__lags
                predict.append(pre)
            self.__ar_test_data = test_data[self.__lags:]
            self.__ar_test_predict = pandas.DataFrame({ 't': predict[self.__lags:]}, index = test_data[self.__lags:].index)
            self.__ar_test_error[0] = Math.mean_square_error(test_data[self.__lags:], self.__ar_test_predict['t'])
            self.__ar_test_error[1] = Math.mean_absolute_error(test_data[self.__lags:], self.__ar_test_predict['t'])
            self.__ar_test_error[2] = Math.root_mean_squard_error(test_data[self.__lags:], self.__ar_test_predict['t'])
            self.__ar_test_error[3] = Math.normalized_mean_squard_error(test_data[self.__lags:], self.__ar_test_predict['t'])
        elif(pure_test_set_predict == False):
            predict = Data.dataframe_to_list(self.__ar_lags_data['t'][-self.__lags:])
            for day in range(len(test_data)):
                pre =  self.__ar_coef_matrix[0][0]
                for i in range(self.__lags, 0, -1):
                    pre += self.__ar_coef_matrix[i][0]*predict[day]
                    day += 1
                day -= self.__lags
                predict.append(pre)
            self.__ar_test_data = test_data
            self.__ar_test_predict = pandas.DataFrame({ 't': predict[self.__lags:]}, index = test_data.index)
            self.__ar_test_error[0] = Math.mean_square_error(test_data, self.__ar_test_predict['t'])
            self.__ar_test_error[1] = Math.mean_absolute_error(test_data, self.__ar_test_predict['t'])
            self.__ar_test_error[2] = Math.root_mean_squard_error(test_data, self.__ar_test_predict['t'])
            self.__ar_test_error[3] = Math.normalized_mean_squard_error(test_data, self.__ar_test_predict['t'])

class SimpleMovingAverage:

    """Simple moving average is often used in the financial field.

    The calculation method is to calculate the average after adding n previous data.

    Parameters
    ---------
        windows: int, default = 5
        Number of lagging periods. Decided to use data from previous periods as explanatory variables.
        The default value is 5, which means that only the data of the previous period is used as an explanatory variable, which is called the simple moving average (SMA(5)).

    Attributes
    ---------
        __data: pandas.Series
        One-dimensional time series data.
        For example:
            date
            2020-06-23 01:00:00    28.89
            2020-06-24 01:00:00    28.91
            2020-06-29 01:00:00    28.87
            2020-06-30 01:00:00    28.92
            2020-07-01 01:00:00    28.79
            Name: close, Length: 170, dtype: float64

        __windows: int, default = 5
        Determine the number of windows containing n lag periods.

        __sma_data: One-dimensional array
        Simple moving average data.
        For example:
            date
            2020-09-29 23:00:00    29.10
            2020-09-30 23:00:00    28.89
            2020-10-01 23:00:00    28.91
            2020-10-04 23:00:00    28.87
            2020-10-06 13:56:21    28.92
            Name: 0, dtype: float64

    Methods
    ---------
        data -> pandas.Series:
            Return the one-dimensional time series data.

        sma_result -> pandas.Series
            Return the result of a one-dimensional number list after simple moving average calculation.

        fit(train_data: pandas.Series)
            Fit simple moving average.

    Examples
    ---------
        import TimeSeriesAnalysis.ProcessData as Data
        import TimeSeriesAnalysis.Model as model

        col_name = 'close'
        file_path = 'Stock.csv'
        times_data_id = 'Stock.TW'

        data = Data.read_file(file_path, col_name)

        model = Model.SimpleMovingAverage(windows= 5)
        model.fit(data)
        print(model.sma_data)

    References
    ---------
    Simple moving average: https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average

    """

    def __init__(self, windows: int = 5):
        self.__data = []
        self.__windows = windows
        self.__sma_result = []

    @property
    def data(self) -> pandas.Series:
        """Return the one-dimensional time series data."""
        return self.__data

    @property
    def sma_result(self) -> pandas.Series:
        """Return the result of a one-dimensional number list after simple moving average calculation."""
        return self.__sma_result[0]

    def __cal_sma_result(self):
        """Calculate simple moving average."""
        if(self.__windows < 2 or self.__windows > len(self.__data)):
            raise ValueError("'windows' > 2 and 'windows' <= data.lenght")
        sma_d = []
        for i in range(0,len(self.__data)-self.__windows+1):
            Sum = 0
            for w in range(self.__windows):
                Sum += self.__data[i+w]
            sma_d.append(Sum/self.__windows)
        self.__sma_result = pandas.DataFrame(sma_d, index = self.__data[self.__windows-1:].index)

    def fit(self, data: pandas.Series):
        """Fit simple moving average.

        Parameters
        ---------
            data: pandas.Series.
            Calculate simple moving average.'data' is a one-dimensional numerical data of type'pandas.Series' taken from'pandas.DataFrame'.

        Returns
        ---------
        No return value.

        Error
        ---------
            ValueError: The type of one-dimensional data must be 'pandas.Series', and the content is numeric data.
            Solution: Check whether the parameter matches the 'ValueError' description, or use the default value.

            ValueError: The list must be one-dimensional numerical data, and there is at least one numerical data in it.
            Solution: The type of one-dimensional data must be 'pandas.Series', and the content is numeric data.

            ValueError: 'windows' > 2 and 'windows' <= data.lenght
            Solution: Check whether the parameter matches the 'ValueError' description, or use the default value.

            TypeError: object of type 'NoneType' has no len().
            Solution:  Please do not enter 'None'. The type of one-dimensional data must be 'pandas.Series', and the content is numeric data.

            KeyError: The 'train_data' type is'pandas.DataFrame', please enter the correct key value.
            Solution: Use the correct key value to obtain data.

        """
        self.__data = data
        self.__cal_sma_result()
