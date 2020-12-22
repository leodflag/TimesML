#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2020 by leodflag

"""Time series data related drawing function.

The drawing function of this module enables time series data to be presented in a 
chart and automatically saved under the folder name entered when creating the category.

Class
---------
chart
    When creating objects of this category, you can define content settings such as chart title, font, and chart size.

Methods
---------
historocal_trend_line_chart(data, xlabel: str = 'date', ylabel: str = 'price')
    Draw historical trend line chart with time series data.

lag_plot(data: pandas.Series)
    Scatter plot of lagging periods.

ACF_chart(data: pandas.Series, lags: int = 1)
    Autocorrelation coefficient chart.

forecast_result_line_graph(test_data, prediction_data, model_name: str = "model")
    Forecast result graph.

Notes
-----------
Pay attention to the input data type of the function.

Use the function "list_to_dataframe" or "dataframe_to_list" of the module "TimeSeriesAnalysis.ProcessData" to convert the data type.

"""
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import Math.Statistics as Math
import TimeSeriesAnalysis.ProcessData as Data

class chart:

    """Time series chart.

    Parameters
    ---------
        title_stock: string, default = 'stock'
        Titles of all time series charts.

        figsize_length: int, default = 20
        The length of the chart.

        figsize_width: int, default = 9
        The width of the chart.

        fontsize_title: int, default = 20
        Font size of chart title.

        fontsize_x_y: int, default = 10
        The font size of the chart x-axis and y-axis scale.

        fontsize_x_y_label: int, default = 13
        The font size of the chart's x-axis and y-axis names.

    Methods
    ---------
        historocal_trend_line_chart(data, xlabel: str = 'date', ylabel: str = 'price')
            Draw historical trend line chart with time series data.

        lag_plot(data: pandas.Series)
            Scatter plot of lagging periods.

        ACF_chart(data: pandas.Series, lags: int = 1)
            Autocorrelation coefficient chart.

        forecast_result_line_graph(test_data, prediction_data, model_name: str = "model")
            Forecast result graph.

    Examples
    ---------
        import TimeSeriesAnalysis.ProcessData as Data
        import TimeSeriesAnalysis.Chart as Chart
        import TimeSeriesAnalysis.Model as model

        col_name = 'close'
        file_path = 'Stock.csv'
        stock_id = 'Stock.TW'

        data = Data.read_file(file_path, col_name)

        chart = Chart.chart(stock_id)
        chart.historocal_trend_line_chart(data)
        chart.lag_plot(data)
        chart.ACF_chart(data, 8)

        train, test = Data.split_data(data, 0.7)
        model = model.AutoRegressive()
        model.fit(train)
        model.predict(test)

        chart.forecast_result_line_graph(model.test_data, model.prediction, 'AR')

    """

    def __init__(self, title_stock: str = 'stock', figsize_length: int = 20, figsize_width: int = 9,
                 fontsize_title: int = 20, fontsize_x_y: int = 10, fontsize_x_y_label: int = 13):
        self.__title_stock = title_stock
        self.__figsize = (figsize_length, figsize_width)
        self.__fontsize_title = fontsize_title
        self.__fontsize_x_y = fontsize_x_y
        self.__fontsize_x_y_label = fontsize_x_y_label

    def historocal_trend_line_chart(self, data, xlabel: str = 'date', ylabel: str = 'price'):
        """Draw historical trend line chart with time series data. The chart will be automatically saved in a folder named 
'title_stock'. Refer to the constructor parameter 'title_stock'.

        Parameters
        ---------
            data: list ,ndarray, pandas.Series and pandas.DataFrame.
            One-dimensional numerical list.

            xlabel: str, default  = 'date'
            The name of the x-axis.

            ylabel: str, default  = 'price'
            The name of the y-axis

        Returns
        ---------
        No return value.

        Error
        ---------
            ValueError: ('List content must be one numerical data: data=', False)
            Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
            non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_data' is'True', it means that 'data' does not need to be 
            changed; if it is'False', the input 'data' is changed to a one-dimensional list of numerical data.

            ValueError: The list must be one-dimensional numerical data, and there is at least one numerical data in it.
            Solution: The entered'listA' is an empty list. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

            KeyError: "The'data' type is'pandas.DataFrame', please enter the correct key value."
            Solution: Use the correct key value to obtain data.

        """
        try:
            chack_data = Math.chack_list_all_num(data)
            if(chack_data):
                plt.figure(figsize = self.__figsize)
                plt.title(" Historocal trend : " + self.__title_stock, fontsize=self.__fontsize_title)
                plt.xlabel(xlabel, fontsize = self.__fontsize_x_y_label)
                plt.ylabel(ylabel, fontsize = self.__fontsize_x_y_label)
                plt.tick_params(axis = 'both', labelsize = self.__fontsize_x_y, color = 'green')
                plt.xticks(rotation=20)
                plt.plot(data)
                if not os.path.isdir(self.__title_stock + '/'):
                    os.mkdir(self.__title_stock)
                plt.savefig(self.__title_stock + "/Historocal_Trend_" + self.__title_stock + ".png")
                plt.show()
                print('Saved successfully (historocal trend). File path = ' +
                      self.__title_stock + "/Historocal_Trend_" + self.__title_stock + ".png")
            else:
                raise ValueError("List content must be one numerical data: data=",chack_data)
        except TypeError as err:
            raise TypeError(err)

    def lag_plot(self, data: pandas.Series):
        """Scatter plot of lagging periods.

        The current time series data and the previous time series data are drawn into a scatter diagram to determine the linear relationship.
        Current data is the x-axis, previous data is the y-axis.The chart will be automatically saved in a folder named 
        'title_stock'. Refer to the constructor parameter 'title_stock'.

        Parameters
        ---------
            data: pandas.Series.
            'data' is a one-dimensional numerical time series data of type'pandas.Series' taken from'pandas.DataFrame'.

        Returns
        ---------
        No return value.

        Error
        ---------
            ValueError: The type of one-dimensional data must be 'pandas.Series', and the content is numeric data.
            Solution: Check whether the parameter matches the 'ValueError' description, or use the default value.

            ValueError: The list must be one-dimensional numerical data, and there is at least one numerical data in it.
            Solution: The entered'data' is an empty list. Check that the type of one-dimensional data must be 'pandas.Series', and the content is numeric data.

            ValueError: 'lags' value must be a positive integer, and the condition is greater than 1 and less than the length of the data.
            Solution: Check whether the parameter matches the 'ValueError' description.

            TypeError: object of type 'NoneType' has no len().
            Solution:  Please do not enter 'None'.Check that the type of one-dimensional data must be 'pandas.Series', and the content is numeric data.

        """
        try:
            plt.figure(figsize=self.__figsize)
            lags_data = Data.create_ar_data(data)
            plt.title("Lag Plot : "+self.__title_stock, fontsize=self.__fontsize_title)
            plt.xlabel("y(t)", fontsize=self.__fontsize_x_y_label)
            plt.ylabel("y(t-1)", fontsize=self.__fontsize_x_y_label)
            plt.tick_params(axis='both', labelsize=self.__fontsize_x_y)
            plt.scatter(lags_data['t'], lags_data['t-1'], s = 10)
            if not os.path.isdir(self.__title_stock + '/'):
                os.mkdir(self.__title_stock)
            plt.savefig(self.__title_stock + "/Lag_Plot_" + self.__title_stock + ".png")
            plt.show()
            print('Saved successfully (lag plot). File path = ' +
                  self.__title_stock + "/Lag_Plot_" + self.__title_stock + ".png")
        except AttributeError as err:
            print(err)

    def ACF_chart(self, data: pandas.Series, lags: int = 1):
        """Autocorrelation coefficient chart.

        Compare a time series data with itself. Bar graph. The number of deferral periods is on the x-axis, and ACF is on the y-axis
        The value range of the obtained autocorrelation value R is [-1,1], 1 is the maximum positive correlation 
        value,-1 is the maximum negative correlation value, and 0 is irrelevant. The chart will be automatically saved in a folder named 
        'title_stock'. Refer to the constructor parameter 'title_stock'.

        Parameters
        ---------
            data: pandas.Series.
            'data' is a one-dimensional numerical time series data of type'pandas.Series' taken from'pandas.DataFrame'.

            lags: int, default  = 1
            'lags' represents the number of lagging periods. The default value is 1, which means the data is 1 period behind.

        Returns
        ---------
        No return value.

        Error
        ---------
            ValueError: The type of one-dimensional data must be 'pandas.Series', and the content is numeric data.
            Solution: Check whether the parameter matches the 'ValueError' description, or use the default value.

            ValueError: The list must be one-dimensional numerical data, and there is at least one numerical data in it.
            Solution: The entered'data' is an empty list. Check that the type of one-dimensional data must be 'pandas.Series', and the content is numeric data.

            ValueError: 'lags' value must be a positive integer, and the condition is greater than 1 and less than the length of the data.
            Solution: Check whether the parameter matches the 'ValueError' description.

            TypeError: object of type 'NoneType' has no len().
            Solution:  Please do not enter 'None'.Check that the type of one-dimensional data must be 'pandas.Series', and the content is numeric data.

        """
        try:
            if(lags >= 1 and type(lags) == int and lags < len(data)):
                ar_data = Data.create_ar_data(data, lags)
                acf = []
                for i in range(1,lags+1):
                    acf_value = 0
                    acf_value = Math.ACF(ar_data['t'], ar_data['t-'+str(i)])
                    acf.append(acf_value)
                df_acf = Data.list_to_dataframe(acf)
                plt.figure(figsize=self.__figsize)
                plt.title("ACF : "+self.__title_stock, fontsize=self.__fontsize_title)
                plt.xlabel("lags", fontsize=self.__fontsize_x_y_label)
                plt.ylabel("ACF", fontsize=self.__fontsize_x_y_label)
                plt.tick_params(axis='both', labelsize=self.__fontsize_x_y)
                plt.bar(df_acf.index, df_acf[0])
                plt.show()
            else:
                raise ValueError("'lags' value must be a positive integer, and the condition is greater than 1 and less than the length of the data.")
        except AttributeError as err:
            raise AttributeError(err)

    def forecast_result_line_graph(self, test_data, prediction_data, model_name: str = "model"):
        """Forecast result graph.

        Input test data and predicted results are drawn as a line graph for comparison.

        Parameters
        ---------
            test_data: list ,ndarray, pandas.Series and pandas.DataFrame.
            One-dimensional numerical list.

            prediction_data: list ,ndarray, pandas.Series and pandas.DataFrame.
            One-dimensional numerical list.

            model_name: str, default   = "model"
            The name of the time series model.

        Returns
        ---------
        No return value.

        Error
        ---------
            ValueError: The two lists must be the same length.
            Solution: Make sure that the number of numerical data in the two lists is the same (the same length).

            ValueError: List content must be one-dimensional numerical data:  test_data=", chack_test, "; prediction_data=", chack_pre.
            Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
            non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_test' is'True', it means that 'test_data' does not need 
            to be changed; if it is'False', the input 'test_data' is changed to a one-dimensional list of numerical data. prediction_data has 
            the same judgment and processing method as test_data.

            TypeError: object of type 'NoneType' has no len()
            Solution: 'listA' or 'listB' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

        """
        try:
            chack_test = Math.chack_list_all_num(test_data)
            chack_pre = Math.chack_list_all_num(prediction_data)
            if(chack_test and chack_pre):
                if (len(test_data) == len(prediction_data)):
                    plt.figure(figsize=(20, 8))
                    plt.title("Forecast Result (" +  model_name + "): " + self.__title_stock , fontsize=20)
                    plt.plot(test_data, label="Test")
                    plt.plot(prediction_data, color="red", label="Predicted")
                    plt.xlabel("Date", fontsize=13)
                    plt.ylabel("Price", fontsize=13)
                    plt.xticks(rotation=45)
                    plt.tick_params(axis='both', labelsize=7)
                    plt.legend(loc='best')
                    if not os.path.isdir(self.__title_stock + '/'):
                        os.mkdir(self.__title_stock)
                    plt.savefig(self.__title_stock + "/Forecast_Result_"+  model_name +"_" +  self.__title_stock + ".png")
                    plt.show()
                    print('Saved successfully  (forecast result '+ model_name +')')
                else:
                    raise ValueError("The two lists must be the same length.")
            else:
                raise ValueError("List content must be one-dimensional numerical data: test_data=", chack_test, "; prediction_data=", chack_pre)
        except TypeError as err:
            raise TypeError(err)
