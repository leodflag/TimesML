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
MovingAverage
	Moving average model is a time series model.

Methods
---------
fit(train_data: pandas.Series)
    Fit moving average model.

predict(test_data)
    Use a moving average model to make predictions.

Notes
-----------
The input data format must be'pandas.Series'.

If the time series data type is a list, you can use the function "list_to_dataframe" of the module "ProcessData" to convert the data type.

It is recommended to use the function of'ProcessData' to obtain and process time series data, which is less prone to errors when entering the model.

"""
import pandas
import numpy
import TimeSeriesAnalysis.ProcessData as Data
import TimeSeriesAnalysis.Chart as Chart
import Math.Statistics as Math

class AutoRegressive:

    """Autoregressive is a time series model.

    Autoregressive is a time series model. The model uses past data in the time series as its 
    own explanatory variable, so it will be affected by current errors and past data.

    Parameters
    ---------
        lags: int, default = 1
        Number of lagging periods. Decide to use the data of the previous periods as the explanatory variable. 
        The default value is 1, which means that only the data from the previous period is used as the explanatory 
        variable, which is called the first order autoregressive model (AR(1)).

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

        __ar_data: pandas.DataFrame
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

        __ar_deviation: float
        The error between the current period and the previous period.

        __ar_coef_matrix: numpy.ndarray
        Multidimensional linear regression matrix normal equation.

        __ar_prediction: list
        Time series forecast results of autoregressive model.

        __ar_mse: float
        The mean square error predicted by the autoregressive model.

    Methods
    ---------
        fit(train_data: pandas.Series)
            Fit autoregression model.

        predict(test_data: pandas.Series) -> list
            Use an autoregressive model to make predictions.

    Examples
    ---------
        import TimeSeriesAnalysis.ProcessData as Data
        import TimeSeriesAnalysis.Model as model

        lags = 1
        col_name = 'close'
        file_path = 'Stock.csv'
        stock_id = 'Stock.TW'

        data = Data.read_file(file_path, col_name)
        train, test = Data.split_data(data, 0.7)

        model = model.AutoRegressive(lags)
        model.fit(train)
        model.predict(test)
        print(model.prediction) # [29.105471220668406, 28.973782041855156, 28.986323868408796, 28.961240215301512, 28.99259478168562]
        print(model.mse) #  0.106772639085492

    References
    ---------
    Autoregressive model: https://zh.wikipedia.org/wiki/%E8%87%AA%E8%BF%B4%E6%AD%B8%E6%A8%A1%E5%9E%8B
    Least Squares Estimation of the Parameters: https://slideplayer.com/slide/4238175/14/images/12/12-1+Multiple+Linear+Regression+Models.jpg
    Matrix Approach to Multiple Linear Regression: https://slideplayer.com/slide/4238175/14/images/20/12-1+Multiple+Linear+Regression+Models.jpg

    """

    def __init__(self, lags: int = 1):
        self.__data = []
        self.__lags = lags
        self.__ar_data = []
        self.__ar_test_data = []
        self.__ar_deviation = 0
        self.__ar_coef_matrix = []
        self.__ar_prediction = []
        self.__ar_mse = 0

    @property
    def data(self) -> pandas.Series:
        """Return the one-dimensional time series data."""
        return self.__data

    @property
    def test_data(self) -> pandas.Series:
        """Return the one-dimensional predicted label data."""
        return self.__ar_test_data['t']

    @property
    def deviation(self):
        return self.__ar_deviation

    @property
    def coefficients_matrix(self) -> numpy.ndarray:
        """Return the autoregressive model coefficients matrix. """
        return self.__ar_coef_matrix

    @property
    def prediction(self) -> list:
        """Return the autoregressive model prediction results."""
        return self.__ar_prediction

    @property
    def mse(self) -> float:
        """Return the mean square error predicted by the autoregressive model."""
        return self.__ar_mse

    def __cal_deviation(self):
        """ Calculate the error between the current period and the previous period."""
        try:
            lenght = len(self.__ar_data) - 1
            dev = self.__ar_data['t'][lenght] - self.__ar_data['t-1'][lenght]
            return dev
        except ValueError as err:
            raise ValueError(err)

    def __cal_ar_normal_equation(self):
        """Multidimensional linear regression matrix normal equation."""
        leng = self.__lags + 1
        A_matrix = numpy.zeros([leng, leng])
        B_matrix = numpy.zeros([leng, 1])
        for i in range(leng):
            for j in range(leng):
                if (i == 0 and j == 0):
                    A_matrix[i][j] = len(self.__ar_data['t'])
                    B_matrix[i][0] = self.__ar_data['t'].sum()
                elif (i == 0 and i != j):
                    A_matrix[i][j] = self.__ar_data['t-' + str(j)].sum()
                    B_matrix[j][0] = (self.__ar_data['t']*self.__ar_data['t-' + str(j)]).sum()
                elif (j == 0 and j !=i ):
                    A_matrix[i][j] = self.__ar_data['t-' + str(i)].sum()
                else:
                    A_matrix[i][j] = (self.__ar_data['t-'+str(i)] * self.__ar_data['t-'+str(j)]).sum()
        A_inv = numpy.linalg.inv(A_matrix)
        coef_matrix = A_inv.dot(B_matrix)
        return coef_matrix

    def fit(self, train_data: pandas.Series):
        """Fit autoregression model.

        Parameters
        ---------
            train_data: pandas.Series.
            Data for training the autoregression model.'train_data' is a one-dimensional numerical data of 
            type'pandas.Series' taken from'pandas.DataFrame'.

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

            KeyError: The 'train_data' type is'pandas.DataFrame', please enter the correct key value.
            Solution: Use the correct key value to obtain data.

        """
        self.__data = train_data
        self.__ar_data = Data.create_ar_data(self.__data, self.__lags)
        self.__ar_deviation = self.__cal_deviation()
        self.__ar_coef_matrix = self.__cal_ar_normal_equation()

    def predict(self, test_data: pandas.Series) -> list:
        """Use an autoregressive model to make predictions.

        Parameters
        ---------
            test_data: pandas.Series.
            Data for testing the autoregression model.'test_data' is a one-dimensional numerical data of 
            type'pandas.Series' taken from'pandas.DataFrame'.

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
        self.__ar_test_data = Data.create_ar_data(test_data, self.__lags)
        for day in range(len(self.__ar_test_data['t'])):
            pre = self.__ar_coef_matrix[0] + self.__ar_deviation
            for i in range(1, len(self.__ar_coef_matrix)):
                pre += self.__ar_coef_matrix[i][0]*self.__ar_test_data['t-'+str(i)][day]
            self.__ar_prediction.append(pre[0])
        self.__ar_mse = Math.mean_square_error(self.__ar_test_data['t'], self.__ar_prediction)

class MovingAverage:

    """Moving average model is a time series model.

    Moving average model refers to the relationship between variables, current errors and previous errors.

    Parameters
    ---------
        lags: int, default = 1
        Number of lagging periods. Decided to use data from previous periods as explanatory variables.
        The default value is 1, which means that only the data of the previous period is used as an 
        explanatory variable, which is called the first-order moving average model (MA(1)).

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

        __ma_data: pandas.DataFrame
        Moving average data set.
        For example MA(1) data set:
            date　　　　　　　　t-1　　t　　Xt
            2020-09-30 23:00:00  0.21 -0.02  28.91
            2020-10-01 23:00:00 -0.02  0.04  28.87
            2020-10-04 23:00:00  0.04 -0.05  28.92
            2020-10-06 13:56:21 -0.05  0.13  28.79
            -------------------------------------------
            't': Current time series data error
            't-1': Previous time series data error
            'Xt':  Current time series data

        __ma_coef_matrix: Two-dimensional array
        Multidimensional linear regression matrix normal equation.

        __ma_prediction: list
        Time series forecast results of moving average model.

        __ma_mse: float
        The mean square error predicted by the moving average model.

    Methods
    ---------
        fit(train_data: pandas.Series)
            Fit moving average model.

        predict(test_data)
            Use a moving average model to make predictions.

    Examples
    ---------
        import TimeSeriesAnalysis.ProcessData as Data
        import TimeSeriesAnalysis.Model as model

        lags = 1
        col_name = 'close'
        file_path = 'Stock.csv'
        stock_id = 'Stock.TW'

        data = Data.read_file(file_path, col_name)
        train, test = Data.split_data(data, 0.7)

        model = model.MovingAverage(lags)
        model.fit(train)
        model.predict(test)
        print(model.prediction) # [28.96937746445215, 29.18523001553352, 29.0545728282949, 29.29555860915283]
        print(model.mse) # 0.37659499938186347

    References
    ---------
    Moving average model: https://zh.wikipedia.org/wiki/%E7%A7%BB%E5%8A%A8%E5%B9%B3%E5%9D%87%E6%A8%A1%E5%9E%8B
    Least Squares Estimation of the Parameters: https://slideplayer.com/slide/4238175/14/images/12/12-1+Multiple+Linear+Regression+Models.jpg
    Matrix Approach to Multiple Linear Regression: https://slideplayer.com/slide/4238175/14/images/20/12-1+Multiple+Linear+Regression+Models.jpg

    """

    def __init__(self, lags: int = 1):
        self.__data = []
        self.__lags = lags
        self.__ma_data = []
        self.__ma_test_data = []
        self.__ma_coef_matrix = []
        self.__ma_prediction = []
        self.__ma_mse = 0

    @property
    def data(self) -> pandas.Series:
        """Return the one-dimensional time series data."""
        return self.__data

    @property
    def test_data(self) -> pandas.Series:
        """Return the one-dimensional predicted label data."""
        return self.__ma_test_data['Xt']

    @property
    def coefficients_matrix(self) -> numpy.ndarray:
        """Return the moving average model coefficients matrix. """
        return self.__ma_coef_matrix

    @property
    def prediction(self) -> list:
        """Return the moving average model prediction results."""
        return self.__ma_prediction

    @property
    def mse(self) -> float:
        """Return the mean square error predicted by the moving average model."""
        return self.__ma_mse

    def __cal_ma_normal_equation(self):
        """Multidimensional linear regression matrix normal equation."""
        leng = self.__lags + 1
        A_matrix = numpy.zeros([leng, leng])
        B_matrix = numpy.zeros([leng, 1])
        for i in range(leng):
            for j in range(leng):
                if (i == 0 and j == 0):
                    A_matrix[i][j] = len(self.__ma_data['t'])
                    B_matrix[i][0] = self.__ma_data['Xt'].sum()
                elif (i == 0 and i != j):
                    A_matrix[i][j] = self.__ma_data['t-' + str(j)].sum()
                    B_matrix[j][0] = (self.__ma_data['Xt']*self.__ma_data['t-' + str(j)]).sum()
                elif (j == 0 and j !=i ):
                    A_matrix[i][j] = self.__ma_data['t-' + str(i)].sum()
                else:
                    A_matrix[i][j] = (self.__ma_data['t-'+str(i)] * self.__ma_data['t-'+str(j)]).sum()
        A_inv = numpy.linalg.inv(A_matrix)
        coef_matrix = A_inv.dot(B_matrix)
        return coef_matrix

    def fit(self, train_data: pandas.Series):
        """Fit moving average model.

        Parameters
        ---------
            train_data: pandas.Series.
            Data for training the moving average model.'train_data' is a one-dimensional numerical data of 
            type'pandas.Series' taken from'pandas.DataFrame'.

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

            KeyError: The 'train_data' type is'pandas.DataFrame', please enter the correct key value.
            Solution: Use the correct key value to obtain data.

        """
        self.__data = train_data
        self.__ma_data = Data.create_ma_data(self.__data, self.__lags)
        self.__ma_coef_matrix = self.__cal_ma_normal_equation()

    def predict(self, test_data):
        """Use a moving average model to make predictions.

        Parameters
        ---------
            test_data: pandas.Series.
            Data for testing the moving average model.'test_data' is a one-dimensional numerical data of 
            type'pandas.Series' taken from'pandas.DataFrame'.

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
        self.__ma_test_data = Data.create_ma_data(test_data, self.__lags)
        for day in range(len(self.__ma_test_data['t'])):
            pre = self.__ma_coef_matrix[0] + self.__ma_test_data['t'][day]
            for i in range(1, len(self.__ma_coef_matrix)):
                pre += self.__ma_coef_matrix[i][0] * self.__ma_test_data['t-'+str(i)][day]
            self.__ma_prediction.append(pre[0])
        self.__ma_mse = Math.mean_square_error( self.__ma_test_data['Xt'], self.__ma_prediction)
