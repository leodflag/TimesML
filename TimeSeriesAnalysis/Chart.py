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
historocal_trend_line_chart(self, data, file_path: str, xlabel: str = 'date', ylabel: str = 'price')
    Draw historical trend line chart with time series data. After the drawing is completed, it will be saved to the specified path 'file_path'.

line_chart(self, data, predict_data, chart_title: str, file_path: str, xlabel: str = 'date', ylabel: str = 'price')
    Throw in two lists of one-dimensional numbers and draw a line chart, it will be saved to the specified path 'file_path'.

lag_plot(data: pandas.Series, file_path: str)
    Scatter plot of lagging periods.

ACF_chart(data: pandas.Series, file_path: str, lags: int = 1)
    Autocorrelation coefficient chart.

forecast_result_chart_model(self, data, model: Model, file_path: str, model_name: str = "model", xlabel='date', ylabel='price')
    The overall prediction results of a model using the training set and the test set.

forecast_result_chart_predict(self, model: Model, file_path: str, model_name: str = "model", xlabel='date', ylabel='price')
    The prediction result of a model using the test set.

statistics_infographic(self, data: pandas.Series, file_path: str, lags: int = 1, xlabel: str = 'date', ylabel: str = 'price')
    Statistical graphs of time series data.

forecast_result_group_chart(self,train, test, model_1: Model, model_2: Model, file_path: str, model_1_name: str = "model1", model_2_name: str = "model2", xlabel: str = 'date', ylabel: str = 'price')
    Combine and compare the prediction results of the two models.

Notes
-----------
Pay attention to the input data type of the function.

Use the function "list_to_dataframe" or "dataframe_to_list" of the module "TimeSeriesAnalysis.ProcessData" to convert the data type.

"""
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import Math.Statistics as Math
import TimeSeriesAnalysis.ProcessData as Data
import TimeSeriesAnalysis.Model as Model

class chart:

    """Time series chart.

    Parameters
    ---------
        title_times_data: string, default = 'times_data'
        Titles of all time series charts.

    Attributes
    ---------
        __figsize_length: int, default = 9
        The length of the chart.

        __figsize_width: int, default = 6
        The width of the chart.

        __fontsize_title: int, default = 15
        Font size of chart title.

        __fontsize_x_y: int, default = 8
        The font size of the chart x-axis and y-axis scale.

        __fontsize_x_y_label: int, default = 11
        The font size of the chart's x-axis and y-axis names.

        self.__data_lag_R: list, default = [0,0]
        The list contains correlation coefficient and coefficient of determination.

        self.__model1_result: list, default = [0,0,0, 0]
        The list contains mean_square_error、mean_absolute_error、root_mean_squard_error and normalized_mean_squard_error.

        self.__model2_result: list, default = [0, 0, 0, 0]
        The list contains mean_square_error、mean_absolute_error、root_mean_squard_error and normalized_mean_squard_error.

    Methods
    ---------

        lag_plot_R_R_square -> dict
            Return correlation coefficient and coefficient of determination.

        forecast_result_group_model1_evaluation-> dict
            Return model1 mse、rmse、nrmse.

        forecast_result_group_model1_evaluation -> dict
            Return model1 mse、rmse、nrmse.

        n_order_difference_graph(self, data, file_path: str, periods: str = 'First', log: bool = False, xlabel: str = 'date')
            Draw n-order difference graph with time series data.

        historocal_trend_line_chart(self, data, file_path: str, xlabel: str = 'date', ylabel: str = 'price')
            Draw historical trend line chart with time series data. After the drawing is completed, it will be saved to the specified path 'file_path'.

        line_chart(self, data, predict_data, chart_title: str, file_path: str, xlabel: str = 'date', ylabel: str = 'price')
            Throw in two lists of one-dimensional numbers and draw a line chart, it will be saved to the specified path 'file_path'.

        lag_plot(data: pandas.Series, file_path: str)
            Scatter plot of lagging periods.

        ACF_chart(data: pandas.Series, file_path: str, lags: int = 1)
            Autocorrelation coefficient chart.

        forecast_result_chart_model(self, data, model: Model, file_path: str, model_name: str = "model", xlabel='date', ylabel='price')
            The overall prediction results of a model using the training set and the test set.

        forecast_result_chart_predict(self, model: Model, file_path: str, model_name: str = "model", xlabel='date', ylabel='price')
            The prediction result of a model using the test set.

        statistics_infographic(self, data: pandas.Series, file_path: str, lags: int = 1, xlabel: str = 'date', ylabel: str = 'price')
            Statistical graphs of time series data.

        forecast_result_group_chart(self,train, test, model_1: Model, model_2: Model, file_path: str, model_1_name: str = "model1", model_2_name: str = "model2", xlabel: str = 'date', ylabel: str = 'price')
            Combine and compare the prediction results of the two models.

    Examples
    ---------
        import TimeSeriesAnalysis.ProcessData as Data
        import TimeSeriesAnalysis.Chart as Chart
        import TimeSeriesAnalysis.Model as model

        col_name = 'close'
        file_path = 'Stock.csv'
        times_data_id = 'Stock.TW'
        save_path = 'Stock_file'

        data = Data.read_file(file_path, col_name)

        chart = Chart.chart(times_data_id)
        chart.historocal_trend_line_chart(data, save_path)
        chart.lag_plot(data, save_path)
        chart.ACF_chart(data, save_path, 8)

        train, test = Data.split_data(data, 0.7)
        model = model.AutoRegressive()
        model.fit(train)
        model.predict(test)

        chart.forecast_result_line_graph(model.test_data, model.prediction, save_path, 'AR')

    """

    def __init__(self, title_times_data: str = 'times_data', figsize_length: int = 9, figsize_width: int = 6,
                 fontsize_title: int = 15, fontsize_x_y: int = 8, fontsize_x_y_label: int =11):
        self.__title_times_data = title_times_data
        self.__figsize = (figsize_length, figsize_width)
        self.__fontsize_title = fontsize_title
        self.__fontsize_x_y = fontsize_x_y
        self.__fontsize_x_y_label = fontsize_x_y_label
        self.__data_lag_R = [0,0]
        self.__model1_result = [0,0,0, 0]
        self.__model2_result = [0, 0, 0, 0]

    @property
    def lag_plot_R_R_square(self) -> dict:
        """Return correlation coefficient and coefficient of determination."""
        dic_lag = {'R': self.__data_lag_R[0], 'R_square': self.__data_lag_R[1]}
        return dic_lag

    @property
    def forecast_result_group_model1_evaluation(self) -> dict:
        """Return model1 mse、rmse、nrmse."""
        dic_forecast = {'mse': self.__model1_result[0], 'mae': self.__model1_result[1],  'rmse': self.__model1_result[2], 'nrmse': self.__model1_result[3]}
        return dic_forecast

    @property
    def forecast_result_group_model2_evaluation(self) -> dict:
        """Return model2 mse、rmse、nrmse."""
        dic_forecast = {'mse': self.__model2_result[0], 'mae': self.__model2_result[1], 'rmse': self.__model2_result[2],  'nrmse': self.__model2_result[3]}
        return dic_forecast

    def n_order_difference_graph(self, data, file_path: str, periods: str = 'First', log: bool = False, xlabel: str = 'date'):
        """Draw n-order difference graph  with time series data.

        For example: First-order difference graph or Second-order difference graph. After the drawing is completed, it will be saved to the specified path 'file_path'.

        Parameters
        ---------
            data: list ,ndarray, pandas.Series and pandas.DataFrame.
            One-dimensional numerical list.

            file_path: string
            Path to save image file.

            periods: str, default  = 'First'
            Differential period.

            log: bool, default  = False
            The data multiplied by the natural logarithm. True or False.

            xlabel: str, default  = 'date'
            The name of the x-axis.

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
            str_log = ''
            if (log == True):
                str_log = 'log'
            if(chack_data):
                plt.figure(figsize = self.__figsize)
                plt.title(periods + " order difference " + str_log + " : "+ self.__title_times_data, fontsize=self.__fontsize_title)
                plt.xlabel(xlabel, fontsize = self.__fontsize_x_y_label)
                plt.xticks(np.linspace(0, len(data), 12), rotation=15)
                plt.plot(data)
                plt.tight_layout()
                if not os.path.isdir(file_path + '/'):
                    os.mkdir(file_path)
                plt.savefig(file_path + "/" + periods + " order difference" + str_log + " : "+ self.__title_times_data + ".png")
                plt.show()
                print('Saved successfully (n order difference). File path = ' +
                      file_path + "/" + periods + "_order_difference_" + str_log +'_'+ self.__title_times_data + ".png")
            else:
                raise ValueError("List content must be one numerical data: data=",chack_data)
        except TypeError as err:
            raise TypeError(err)

    def historocal_trend_line_chart(self, data, file_path: str, xlabel: str = 'date', ylabel: str = 'price'):
        """Draw historical trend line chart with time series data. After the drawing is completed, 
        it will be saved to the specified path 'file_path'.

        Parameters
        ---------
            data: list ,ndarray, pandas.Series and pandas.DataFrame.
            One-dimensional numerical list.

            file_path: string
            Path to save image file.

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
                plt.title(" Historocal trend : " + self.__title_times_data, fontsize=self.__fontsize_title)
                plt.xlabel(xlabel, fontsize = self.__fontsize_x_y_label)
                plt.ylabel(ylabel, fontsize = self.__fontsize_x_y_label)
                plt.tick_params(axis = 'both', labelsize = self.__fontsize_x_y)
                plt.xticks(np.linspace(0, len(data), 12), rotation=15)
                plt.plot(data)
                plt.tight_layout()
                if not os.path.isdir(file_path + '/'):
                    os.mkdir(file_path)
                plt.savefig(file_path + "/Historocal_Trend_" + self.__title_times_data + ".png")
                plt.show()
                print('Saved successfully (historocal trend). File path = ' +
                      file_path + "/Historocal_Trend_" + self.__title_times_data + ".png")
            else:
                raise ValueError("List content must be one numerical data: data=",chack_data)
        except TypeError as err:
            raise TypeError(err)

    def line_chart(self, data, predict_data, chart_title: str, file_path: str, xlabel: str = 'date', ylabel: str = 'price'):
        """Throw in two lists of one-dimensional numbers and draw a line chart, it will be saved to the specified path 'file_path'.

        Parameters
        ---------
            data: list ,ndarray, pandas.Series and pandas.DataFrame.
            One-dimensional numerical list.

            predict_data: list ,ndarray, pandas.Series and pandas.DataFrame.
            One-dimensional numerical list.

            chart_title: string
            Chart title.

            file_path: string
            Path to save image file.

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
            chack_pre = Math.chack_list_all_num(predict_data)
            if(chack_data and chack_pre):
                plt.figure(figsize = self.__figsize)
                plt.title(chart_title + ' :' + self.__title_times_data, fontsize=self.__fontsize_title)
                plt.xlabel(xlabel, fontsize = self.__fontsize_x_y_label)
                plt.ylabel(ylabel, fontsize = self.__fontsize_x_y_label)
                plt.tick_params(axis = 'both', labelsize = self.__fontsize_x_y)
                plt.xticks(np.linspace(0, len(data), 12), rotation=15)
                plt.plot(data, color='lightgray')
                plt.plot(predict_data, color = 'green')
                plt.tight_layout()
                if not os.path.isdir(file_path + '/'):
                    os.mkdir(file_path)
                plt.savefig(file_path + "/" + chart_title + "_" + self.__title_times_data + "_line_chart.png")
                plt.show()
                print('Saved successfully (historocal trend analysis). File path = ' +
                      file_path + "/" + chart_title + "_" + self.__title_times_data + "_line_chart.png")
            else:
                raise ValueError("List content must be one numerical data: data=",chack_data, " predict_data=", chack_pre)
        except TypeError as err:
            raise TypeError(err)

    def lag_plot(self, data: pandas.Series, file_path: str):
        """Scatter plot of lagging periods.

        The current time series data and the previous time series data are drawn into a scatter diagram to determine the linear relationship.
        Current data is the x-axis, previous data is the y-axis.After the drawing is completed, it will be saved to the specified path 'file_path'.

        Parameters
        ---------
            data: pandas.Series.
            'data' is a one-dimensional numerical time series data of type'pandas.Series' taken from'pandas.DataFrame'.

            file_path: string
            Path to save image file.

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
            x_site = min(lags_data['t'])
            y_site = min(lags_data['t-1'])
            self.__data_lag_R[0] = round(Math.correlation_coefficient(lags_data['t'], lags_data['t-1']), 4)
            self.__data_lag_R[1] = round(Math.coefficient_of_determination(lags_data['t'], lags_data['t-1']), 4)
            plt.title("Lag Plot : "+self.__title_times_data, fontsize=self.__fontsize_title)
            plt.xlabel("y(t)", fontsize=self.__fontsize_x_y_label)
            plt.ylabel("y(t-1)", fontsize=self.__fontsize_x_y_label)
            plt.tick_params(axis='both', labelsize=self.__fontsize_x_y)
            plt.text(x_site, y_site, r'$\mathrm{r} = $' + str(self.__data_lag_R[0]) + '\n' +
                                                          r'$\mathrm{r}^{2} = $' + str(self.__data_lag_R[1]),
                     bbox=dict( boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.9, 0.9), ), fontsize=self.__fontsize_x_y_label)
            plt.scatter(lags_data['t'], lags_data['t-1'], s = 10)
            plt.tight_layout()
            if not os.path.isdir(file_path + '/'):
                os.mkdir(file_path)
            plt.savefig(file_path + "/Lag_Plot_" + self.__title_times_data + ".png")
            plt.show()
            print('Saved successfully (lag plot). File path = ' + file_path + "/Lag_Plot_" + self.__title_times_data + ".png")
        except AttributeError as err:
            print(err)

    def ACF_chart(self, data: pandas.Series, file_path: str, lags: int = 1):
        """Autocorrelation coefficient chart.

        Compare a time series data with itself. Bar graph. The number of deferral periods is on the x-axis, and ACF is on the y-axis
        The value range of the obtained autocorrelation value R is [-1,1], 1 is the maximum positive correlation 
        value,-1 is the maximum negative correlation value, and 0 is irrelevant. After the drawing is completed, it will be saved to the specified path 'file_path'.

        Parameters
        ---------
            data: pandas.Series
            'data' is a one-dimensional numerical time series data of type'pandas.Series' taken from'pandas.DataFrame'.

            file_path: string
            Path to save image file.

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
                index_list = Data.create_sequence_list(acf)
                df_acf = Data.list_to_dataframe(acf, index_list)
                plt.figure(figsize=self.__figsize)
                plt.title("ACF : "+self.__title_times_data, fontsize=self.__fontsize_title)
                plt.xlabel("lags", fontsize=self.__fontsize_x_y_label)
                plt.ylabel("ACF", fontsize=self.__fontsize_x_y_label)
                plt.tick_params(axis='both', labelsize=self.__fontsize_x_y)
                plt.bar(df_acf.index, df_acf[0])
                plt.tight_layout()
                if not os.path.isdir(file_path + '/'):
                    os.mkdir(file_path)
                plt.savefig(file_path + "/ACF_" + self.__title_times_data + ".png")
                plt.show()
                print('Saved successfully (ACF). File path = ' + file_path + "/ACF_" + self.__title_times_data + ".png")
            else:
                raise ValueError("'lags' value must be a positive integer, and the condition is greater than 1 and less than the length of the data.")
        except AttributeError as err:
            raise AttributeError(err)

    def forecast_result_chart_model(self, data, model: Model, file_path: str, model_name: str = "model", xlabel='date', ylabel='price'):
        """The overall prediction results of a model using the training set and the test set.

        The graph drawn by this function can display historical data, the prediction results of the training set 
        and the test set at the same time, as well as the model measurement indicators, such as MSE, MAE, RMSE, 
        NRMSE, and finally saved in the folder.

        Parameters
        ---------
            data: list ,ndarray, pandas.Series and pandas.DataFrame.
            One-dimensional numerical list.

            model: Model
            The object type is TimeSeriesAnalysis.Model.

            file_path: string
            Path to save image file.

            model_name: str, default  = "model"
            The name of the model.

            xlabel: str, default  = 'date'
            The name of the x-axis.

            ylabel: str, default  = 'price'
            The name of the y-axis

        Returns
        ---------
        No return value.

        Error
        ---------
            TypeError: object of type 'NoneType' has no len().
            Solution:  Please do not enter 'None'.Check that the type of one-dimensional data must be 'pandas.Series', and the content is numeric data.

        """
        try:
            plt.figure(figsize=self.__figsize)
            plt.title("Forecast Result Model (" +  model_name + "): " + self.__title_times_data, fontsize=self.__fontsize_title)
            plt.plot(data, color='lightgray', label='actul')
            plt.plot(model.train_predict, label="train predict")
            plt.plot(model.test_predict, color="red", label="test predict")
            plt.xlabel(xlabel, fontsize=self.__fontsize_x_y_label)
            plt.ylabel(ylabel, fontsize=self.__fontsize_x_y_label)
            x_site = min(data)
            y_site = min(data)
            plt.text(x_site*0.05, y_site, r'$\mathrm{train} : $' + '\n\n' +
                     r'$\mathrm{MSE} = $' + str(round(model.train_error[0], 4)) + '\n' +
                     r'$\mathrm{MAE} = $' + str(round(model.train_error[1], 4)) + '\n' +
                     r'$\mathrm{RMSE} = $' + str(round(model.train_error[2], 4)) + '\n' +
                     r'$\mathrm{NRMSE} = $' + str(round(model.train_error[3], 4)) 
                     + '\n\n' + r'$\mathrm{test} : $' + '\n\n' + r'$\mathrm{MSE} = $' + str(round(model.test_error[0], 4)) 
                      +'\n' + r'$\mathrm{MAE} = $' + str(round(model.test_error[1], 4)) + '\n' + r'$\mathrm{RMSE} = $' + str(round(model.test_error[2], 4)) 
                      + '\n' + r'$\mathrm{NRMSE} = $' + str(round(model.test_error[3], 4)),
                            bbox=dict( boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.9, 0.9), ), fontsize=self.__fontsize_x_y)
            plt.xticks(np.linspace(0, len(data), 12), rotation=15)
            plt.tick_params(axis='both', labelsize=self.__fontsize_x_y)
            plt.legend(loc='upper left')
            plt.tight_layout()
            if not os.path.isdir(file_path + '/'):
                os.mkdir(file_path)
            plt.savefig(file_path + "/Forecast_Result_Model_"+  model_name +"_" +  self.__title_times_data + ".png")
            plt.show()
            print('Saved successfully (forecast result model '+ model_name +'). File path = ' + file_path +
                    "/Forecast_Result_Model" + model_name + "_" + self.__title_times_data + ".png")
        except TypeError as err:
            raise TypeError(err)

    def forecast_result_chart_predict(self, model: Model, file_path: str, model_name: str = "model", xlabel='date', ylabel='price'):
        """Forecast result graph.

        Input test data and predicted results are drawn as a line graph for comparison.After the drawing is completed, it will be saved to the specified path 'file_path'.

        Parameters
        ---------
            model: Model
            The object type is TimeSeriesAnalysis.Model.

            file_path: string
            Path to save image file.

            model_name: str, default   = "model"
            The name of the time series model.

        Returns
        ---------
        No return value.

        Error
        ---------
            ValueError: The two lists must be the same length.
            Solution: Make sure that the number of numerical data in the two lists is the same (the same length).

            ValueError: List content must be one-dimensional numerical data:  test_data=", chack_test, "; predict_data=", chack_pre.
            Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
            non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_test' is'True', it means that 'test_data' does not need 
            to be changed; if it is'False', the input 'test_data' is changed to a one-dimensional list of numerical data. predict_data has 
            the same judgment and processing method as test_data.

            TypeError: object of type 'NoneType' has no len()
            Solution: 'listA' or 'listB' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

        """
        try:
            plt.figure(figsize=self.__figsize)
            plt.title("Forecast Result " +  model_name + " : " + self.__title_times_data, fontsize=self.__fontsize_title)
            plt.plot(model.test_data, color='lightgray', label='actul')
            plt.plot(model.test_predict, color="red", label="test predict")
            plt.xlabel(xlabel, fontsize=self.__fontsize_x_y_label)
            plt.ylabel(ylabel, fontsize=self.__fontsize_x_y_label)
            x_site = len(model.test_data)
            y_site = min(model.test_data)
            plt.text(x_site*0.05, y_site, r'$\mathrm{test} : $' + '\n\n' + r'$\mathrm{MSE} = $' + str(round(model.test_error[0], 4)) 
                      +'\n' + r'$\mathrm{MAE} = $' + str(round(model.test_error[1], 4)) + '\n' + r'$\mathrm{RMSE} = $' + str(round(model.test_error[2], 4)) 
                      + '\n' + r'$\mathrm{NRMSE} = $' + str(round(model.test_error[3], 4)),
                            bbox=dict( boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.9, 0.9), ), fontsize=self.__fontsize_x_y)
            plt.xticks(np.linspace(0, len(model.test_data), 12), rotation=15)
            plt.tick_params(axis='both', labelsize=self.__fontsize_x_y)
            plt.legend(loc='best')
            plt.tight_layout()
            if not os.path.isdir(file_path + '/'):
                os.mkdir(file_path)
            plt.savefig(file_path + "/Forecast_Result_predict_"+  model_name +"_" +  self.__title_times_data + ".png")
            plt.show()
            print('Saved successfully (forecast result presict '+ model_name +'). File path = ' + file_path +
                    "/Forecast_Result_predict_" + model_name + "_" + self.__title_times_data + ".png")
        except TypeError as err:
            raise TypeError(err)

    def statistics_infographic(self, data: pandas.Series, file_path: str, lags: int = 1, xlabel: str = 'date', ylabel: str = 'price'):
        """Statistical graphs of time series data.

    Three statistical graphs combined into one graph: historocal trend line chart、lag plot、ACF chart.

        Parameters
        ---------
            data: pandas.Series
            'data' is a one-dimensional numerical time series data of type'pandas.Series' taken from'pandas.DataFrame'.

            file_path: string
            Path to save image file.

            lags: int, default  = 1
            'lags' represents the number of lagging periods. The default value is 1, which means the data is 1 period behind.

            xlabel: str, default  = 'date'
            The name of the x-axis for historocal trend line chart.

            ylabel: str, default  = 'price'
            The name of the y-axis for historocal trend line chart.

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
                chack_data = Math.chack_list_all_num(data)
                if(chack_data):
                    plt.figure(figsize=(15,9))
                    G = gridspec.GridSpec(2, 4)
                    ar_data = Data.create_ar_data(data, lags)

                    # lag_plot
                    x_site = min(ar_data['t'])
                    y_site = min(ar_data['t-1'])
                    self.__data_lag_R[0] = round(Math.correlation_coefficient(ar_data['t'], ar_data['t-1']), 4)
                    self.__data_lag_R[1] = round(Math.coefficient_of_determination(ar_data['t'], ar_data['t-1']), 4)

                    # ACF_chart
                    acf = []
                    for i in range(1,lags+1):
                        acf_value = 0
                        acf_value = Math.ACF(ar_data['t'], ar_data['t-'+str(i)])
                        acf.append(acf_value)
                    index_list = Data.create_sequence_list(acf)
                    df_acf = Data.list_to_dataframe(acf, index_list)

                    # historocal_trend_line
                    ax1 = plt.subplot(G[0, :])
                    plt.title(" Historocal trend : " + self.__title_times_data, fontsize=self.__fontsize_title)
                    plt.xlabel(xlabel, fontsize = self.__fontsize_x_y_label)
                    plt.ylabel(ylabel, fontsize = self.__fontsize_x_y_label)
                    plt.tick_params(axis = 'both', labelsize = self.__fontsize_x_y, color = 'green')
                    plt.xticks(np.linspace(0, len(data), 12), rotation=15)
                    plt.plot(data)

                    # lag_plot
                    ax2 = plt.subplot(G[1, :2])
                    plt.title("Lag Plot : "+self.__title_times_data, fontsize=self.__fontsize_title)
                    plt.xlabel("y(t)", fontsize=self.__fontsize_x_y_label)
                    plt.ylabel("y(t-1)", fontsize=self.__fontsize_x_y_label)
                    plt.tick_params(axis='both', labelsize=self.__fontsize_x_y)
                    plt.text(x_site, y_site, r'$\mathrm{r} = $' + str(self.__data_lag_R[0]) + '\n' +
                                        r'$\mathrm{r}^{2} = $' + str(self.__data_lag_R[1]),
                            bbox=dict(boxstyle="square",
                                                ec=(1., 0.5, 0.5),
                                                fc=(1., 0.9, 0.9),
                                                ), fontsize=self.__fontsize_x_y_label)
                    plt.scatter(ar_data['t'], ar_data['t-1'], s=10)

                    # ACF_chart
                    ax3 = plt.subplot(G[1, -2:])
                    plt.title("ACF : "+self.__title_times_data, fontsize=self.__fontsize_title)
                    plt.xlabel("lags", fontsize=self.__fontsize_x_y_label)
                    plt.ylabel("ACF", fontsize=self.__fontsize_x_y_label)
                    plt.tick_params(axis='both', labelsize=self.__fontsize_x_y)
                    plt.bar(df_acf.index, df_acf[0])
                    plt.tight_layout()

                    if not os.path.isdir(file_path + '/'):
                        os.mkdir(file_path)
                    plt.savefig(file_path + "/statistics_infographic_" + self.__title_times_data + ".png")
                    plt.show()
                    print('Saved successfully (statistics_infographic). File path = ' + file_path + "/statistics_infographic_" + self.__title_times_data + ".png")

                else:
                    raise ValueError("List content must be one numerical data: data=",chack_data)
            else:
                raise ValueError("'lags' value must be a positive integer, and the condition is greater than 1 and less than the length of the data.")
        except AttributeError as err:
            raise AttributeError(err)
        except TypeError as err:
            raise TypeError(err)

    def forecast_result_group_chart(self,train, test, model_1: Model, model_2: Model, file_path: str, model_1_name: str = "model1", model_2_name: str = "model2", xlabel: str = 'date', ylabel: str = 'price'):
        """Combine and compare the prediction results of the two models.

        The graph drawn by this function can visualize the data divided into the training set and the test set in the same historical data, 
        the prediction results of the test sets of the two models, and the model measurement indicators, such as MSE, MAE, RMSE, NRMSE, finally saved in the folder.

        Parameters
        ---------
            train: pandas.Series
            'train' is a one-dimensional numerical time series data of type'pandas.Series' taken from'pandas.DataFrame'.

            test: pandas.Series
            'test' is a one-dimensional numerical time series data of type'pandas.Series' taken from'pandas.DataFrame'.

            model_1: Model
            The object type is TimeSeriesAnalysis.Model. The First model.

            model_2: Model
            The object type is TimeSeriesAnalysis.Model. The second model.

            file_path: string
            Path to save image file.

            model_1_name: str, default  = "model1"
            The name of the model_1.

            model_2_name: str, default  = "model2"
            The name of the model_2.

            xlabel: str, default  = 'date'
            The name of the x-axis.

            ylabel: str, default  = 'price'
            The name of the y-axis

        Returns
        ---------
        No return value.

        Error
        ---------
            ValueError: List content must be one-dimensional numerical data: train=", chack_train, ", test=", chack_test).
            Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
            non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_train' is'True', it means that 'train' does not need to be 
            changed; if it is'False', the input'train' is changed to a one-dimensional list of numerical data.'test' has the same judgment and processing 
            method as 'train'.

            TypeError: object of type 'NoneType' has no len()
            Solution: 'train' or 'test' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

        """
        try:
            chack_train = Math.chack_list_all_num(train)
            chack_test = Math.chack_list_all_num(test)
            if(chack_train and chack_test):
                plt.figure(figsize=(15,9))
                G = gridspec.GridSpec(2, 4)

                # historocal_trend_line
                ax1 = plt.subplot(G[0, :])
                plt.title(" Historocal trend : " + self.__title_times_data, fontsize=self.__fontsize_title)
                plt.plot(train, color='lightgray', label = 'Train')
                plt.plot(test, color='gold', label = 'Test')
                plt.plot(model_1.test_predict, color='r', label=model_1_name)
                plt.plot(model_2.test_predict, label=model_2_name)
                plt.xlabel(xlabel, fontsize = self.__fontsize_x_y_label)
                plt.ylabel(ylabel, fontsize = self.__fontsize_x_y_label)
                plt.tick_params(axis = 'both', labelsize = self.__fontsize_x_y)
                plt.xticks(np.linspace(0, len(train)+ len(test), 12), rotation=15)
                ax1.legend()

                ax2 = plt.subplot(G[1, :2])
                plt.title("Forecast Result  " +  model_1_name + " : " + self.__title_times_data, fontsize=self.__fontsize_title)
                plt.plot(model_1.test_data, color='lightgray', label="Test")
                plt.plot(model_1.test_predict, color="r", label=model_1_name)
                plt.xlabel(xlabel, fontsize=self.__fontsize_x_y_label)
                plt.ylabel(ylabel, fontsize=self.__fontsize_x_y_label)
                x_site = len(model_1.test_data)
                if(min(model_1.test_data) < min(model_1.test_predict)):
                    y_site = min(model_1.test_data)
                else:
                    y_site = min(model_1.test_predict)
                round_mod_1_err = Math.round_list(model_1.test_error, 4)
                self.__model1_result = round_mod_1_err
                plt.text(x_site*0.05, y_site, r'$\mathrm{MSE} = $' + str(self.__model1_result[0]) + '\n' + r'$\mathrm{MAE} = $' + str(self.__model1_result[1]) +'\n'
                                                                    + r'$\mathrm{RMSE} = $' + str(self.__model1_result[2]) + '\n' + r'$\mathrm{NRMSE} = $' + str(self.__model1_result[3]),
                                    bbox=dict( boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.9, 0.9), ), fontsize=self.__fontsize_x_y)
                plt.xticks(np.linspace(0, len(model_1.test_data), 12), rotation=15)
                plt.tick_params(axis='both', labelsize=self.__fontsize_x_y)
                plt.legend(loc='upper left')

                ax3 = plt.subplot(G[1, -2:])
                plt.title("Forecast Result  " +  model_2_name + " : " + self.__title_times_data, fontsize=self.__fontsize_title)
                plt.plot(model_2.test_data, color='lightgray', label="Test")
                plt.plot(model_2.test_predict, label=model_2_name)
                plt.xlabel(xlabel, fontsize=self.__fontsize_x_y_label)
                plt.ylabel(ylabel, fontsize=self.__fontsize_x_y_label)
                x_site = len(model_2.test_data)
                if(min(model_2.test_data) < min(model_2.test_predict)):
                    y_site = min(model_2.test_data)
                else:
                    y_site = min(model_2.test_predict)
                round_mod_2_err = Math.round_list(model_2.test_error, 4)
                self.__model2_result = round_mod_2_err
                plt.text(x_site*0.05, y_site, r'$\mathrm{MSE} = $' + str(self.__model2_result[0]) + '\n' + r'$\mathrm{MAE} = $' + str(self.__model2_result[1]) +'\n'
                                                                    + r'$\mathrm{RMSE} = $' + str(self.__model2_result[2]) + '\n' + r'$\mathrm{NRMSE} = $' + str(self.__model2_result[3]),
                                    bbox=dict( boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.9, 0.9), ), fontsize=self.__fontsize_x_y)
                plt.xticks(np.linspace(0, len(model_2.test_data), 12), rotation=15)
                plt.tick_params(axis='both', labelsize=self.__fontsize_x_y)
                plt.legend(loc='upper left')
                plt.tight_layout()

                if not os.path.isdir(file_path + '/'):
                    os.mkdir(file_path)
                plt.savefig(file_path + "/forecast_result_group_chart_" + self.__title_times_data + ".png")
                plt.show()
                print('Saved successfully (forecast result group chart). File path = ' + file_path + "/forecast_result_group_chart_" + self.__title_times_data + ".png")
            else:
                raise ValueError("List content must be one-dimensional numerical data: train=", chack_train, ", test=", chack_test)

        except AttributeError as err:
            raise AttributeError(err)
        except TypeError as err:
            raise TypeError(err)
