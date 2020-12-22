#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2020 by leodflag

"""This module contains the access and processing methods of time series data.

Ways to obtain time series data: read files and download from websites.
The data processing method is archiving, the two formats are interchanged, 
the data is divided, and the time series model data set is established.

Functions
---------
read_file(path: str, col_name: str='close') -> pandas.Series
	Read file to get time series data.

get_data_yahoo(stock_id: str, start_period: str, end_period: str, file_format: str='csv', frequency: str = 'day')
	Crawl and archive the stock price data of finance.yahoo.com.

save_flie(data, path: str, stock_id: str = 'stock', file_format: str = 'csv')
	Save data as file.

create_sequence_list(data) -> list
	Create a sequence starting from 1.

list_to_dataframe(data: list)
	Convert the 'list' type of the one-dimensional list of values to the 'DataFrame' type.

dataframe_to_list(data: pandas.Series)
	Convert the 'pandas.Series' type of the one-dimensional list of values to the'list' type.

split_data(data, ratio: float)
	Split the data into two data sets according to the input ratio.

create_ar_data(data: pandas.Series, lags: int = 1) -> pandas.DataFrame
	Create an autoregressive data set.

n_order_difference_data(data: pandas.Series, periods: int = 1, log: bool = False) -> pandas.Series:
    Sequence of numbers after difference operation.

Notes
-----------
If the time series data type is a list, you can use the function "list_to_dataframe" of the module "TimeSeriesAnalysis.ProcessData" to convert the data type.

"""
import requests
import datetime
import json
import time
import io
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse
import Math.Statistics as Math

def read_file(path: str, col_name: str='close') -> pandas.Series:
    """Read file to get time series data.

    The file formats that can be read by this function are csv and txt, and the returned data type is pandas.Series.

    Parameters
    ---------

        path : string
        Enter the absolute path or relative path where the file is located. The file formats that can be read are csv and txt.

        col_name : string , default = 'close'.
        The text of the file must contain column names.

    Returns
    ---------
    pandas.Series. Return a list with date as the column index value, the data type is float.

    Error
    ---------
        FileNotFoundError : [Errno 2] There is no such file or directory: 'wrong path'
        Solution : Enter the correct file path and the correct file name.

        ValueError: Invalid file path or buffer object type: <class 'int'>
        Solution : Enter the correct file path and the correct file name.

        KeyError : "This file has not index value", err.
        Solution : Actually check whether the column name contained in the file content exists, and enter the correct column name.

    """
    try:
        data = pandas.read_csv(path, index_col='date')
        Data = data[col_name]
        return Data
    except (FileNotFoundError, ValueError) as err:
        print(err)
    except KeyError as err:
        print("This file has not index value", err)

def get_data_yahoo(stock_id: str, start_period: str, end_period: str, file_format: str='csv', frequency: str = 'day'):
    """Crawl and archive the stock price data of finance.yahoo.com.

    This function can crawl the stock market price data at a specified time and frequency and save it as a csv or txt file, and the returned 
    data type is pandas.Series.

    Parameters
    ---------
        stock_id: string
        Enter the stock ID. For example: Taiwan stock" 2206.TW"; US stock "SWKS".

        start_period: string
        The start time of the time period for crawling stock prices.  
        For example: "2019 7 12", "2019, 7, 12", "2019 7", "2019, 7", "2019",  "7 21 2019", "7, 21, 2019", " 7 2019","7, 2019".

        end_period: string
        The start time of the time period for crawling stock prices.  
        For example: "2019 8 12", "2019, 8, 12", "2019 8", "2019, 8", "2019",  "8 21 2019", "8, 21, 2019", " 8 2019","8, 2019".

        file_format: string, default = 'csv'.
        Archive the crawled stock market price data, file format is 'csv' or 'txt'.  For example: 'csv' ,'CSV' , 'txt', 'TXT'.

        frequency: string, default = 'day'.
        The frequency of stock market price data is daily, weekly, and monthly. For example: 'daily', 'day', 'weekly', 'week', 'monthly', 'month'.

    Returns
    ---------
    No return.

    Error
    ---------
        ValueError: 'start_period' must be earlier than 'end_period'.
        Solution: Check whether the parameter matches the 'ValueError' description. For example: start_period = '2020, 2, 5',  end_period = '2020, 4, 5'

        TypeError: Not found stock id.
        Solution: Enter the correct stock id.

        ConnectionError: Internet connection error.
        Solution: Confirm that the network is connected.

    """
    try:
        period1 = parse(start_period)
        period2 = parse(end_period)
        if(period1 > period2 or period1 == period2):
            raise ValueError("'start_period' must be earlier than 'end_period'")

        freq = ''
        if(frequency == 'monthly' or frequency == 'month'):
            freq = 'mo'
        elif (frequency == 'weekly' or frequency == 'week'):
            freq = 'wk'
        elif (frequency == 'daily' or frequency == 'day'):
            freq = 'd'
        else:
            freq = 'd'

        url = ("https://query1.finance.yahoo.com/v7/finance/chart/" + stock_id + "?period1=" + str(int(period1.timestamp())) +
               "&period2=" + str(int(period2.timestamp())) + "&interval=1" + freq + "&events=history")
        req = requests.get(url)
        stock = json.loads(req.text)
        if(stock['chart']['result'] == None):
            raise TypeError("Not found stock id")
        else:
            print("Crawl successfully")
            stock_data = pandas.DataFrame(stock['chart']['result'][0]['indicators']['quote'][0], index=pandas.to_datetime(
                np.array(stock['chart']['result'][0]['timestamp'])*1000*1000*1000))
            if not os.path.isdir(stock_id + '/'):
                os.mkdir(stock_id)
            if (file_format == 'csv' or file_format == 'CSV'):
                stock_data.index = stock_data.index.set_names(['date'])
                stock_data.to_csv(stock_id + '/'+stock_id +'_' + frequency + '.csv')
            elif (file_format == 'txt' or file_format == 'TXT'):
                stock_data.index = stock_data.index.set_names(['date'])
                stock_data.to_csv(stock_id + '/'+stock_id + '_' + frequency + '.txt')
    except requests.exceptions.ConnectionError as err:
        raise requests.exceptions.ConnectionError(err)

def save_flie(data, path: str, stock_id: str = 'stock', file_format: str = 'csv'):
    """Save data as file.

    This function can save the input pandas.Series data type data into a csv or txt file.

    Parameters
    ---------
        data: pandas.Series and pandas.DataFrame.
        pandas.Series: One-dimensional ndarray with axis labels (including time series). 
        pandas.DataFrame: Two-dimensional, size-mutable, potentially heterogeneous tabular data.

        path: str.
        Save files to absolute path or relative path.If you enter a null string '' or "",it will be saved in the current folder

        stock_id: str, default = 'stock'.
        File name when saving the file.

        file_format: str, default = 'csv'.
        Archive the crawled stock market price data, file format is 'csv' or 'txt'.  For example: 'csv' ,'CSV' , 'txt', 'TXT'.

    Returns
    ---------
    No return value.

    Error
    ---------
        PermissionError: Reject operations that do not comply with permissions.
        Solution: Path or file name cannot be '/'.

        AttributeError: 'list' object has no attribute 'to_csv'
        Solution: The data format must be pandas.Series and pandas.DataFrame.

        FileNotFoundError: No such file or file path.
        Solution: Enter the correct file path.

    """
    try:
        if(file_format == 'csv' or file_format == 'CSV'):
            data.to_csv(path + stock_id + '.csv')
            print('Saved successfully  (csv)')
        if (file_format == 'txt' or file_format == 'TXT'):
            data.to_csv(path + stock_id + '.txt')
            print('Saved successfully (txt)')
    except (PermissionError, AttributeError, FileNotFoundError) as err:
        print(err)

def create_sequence_list(data) -> list:
    """Create a sequence starting from 1.

    This function uses the length of the input one-dimensional numerical list to create a sequence starting from 1.

    Parameters
    ---------
        data: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    list. Return a sequence list with the same length as 'data'.

    Error
    ---------
        ValueError: List content must be one-dimensional numerical data: chack_data=",chack_data.
        Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
        non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_data' is'True', it means that 'data' does not need to 
        be changed; if it is'False', the input 'data' is changed to a one-dimensional list of numerical data.

        ValueError: 'data' must be one-dimensional numerical list.
        Solution: Check that the input 'data' is a one-dimensional list, for example: [1,5,8,6,3].

    """
    try:
        chack_data = Math.chack_list_all_num(data)
        if(chack_data):
            sequence = []
            data = np.array(data)
            if(data.ndim == 1 and len(data) != 0):
                for i in range(1, (len(data)+1)): sequence.append(i)
                return sequence
        else:
            raise ValueError("List content must be one numerical data: chack_data=", chack_data)
    except :
        raise ValueError("'data' must be one-dimensional numerical list.")

def list_to_dataframe(data: list , index) -> pandas.DataFrame:
    """Convert the 'list' type of the one-dimensional list of values to the 'DataFrame' type.

    Parameters
    ---------
        data: list.
        One-dimensional numerical list. For example: [1,5,8,6,3].

        index: pandas.Index, list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional list.

    Returns
    ---------
    pandas.DataFrame. The one-dimensional list of values to the 'DataFrame' type.

    Error
    ---------
        ValueError: 'data' must be one-dimensional numerical list.
        Solution: 'data' contains a string or is two-dimensional list. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

        ValueError: The two lists must be the same length.
        Solution: Make sure that the number of numerical data in the two lists is the same (the same length).

    """
    try:
        if (len(data) == len(index)):
            df_data = pandas.DataFrame(data, index = index)
            return df_data
        else:
            raise ValueError("The two lists must be the same length.")
    except :
        raise ValueError("'data' must be one-dimensional numerical list.")

def dataframe_to_list(data: pandas.Series) -> list:
    """Convert the 'pandas.Series' type of the one-dimensional list of values to the'list' type.

    Parameters
    ---------
        data: pandas.Series.
        'data' is a one-dimensional numerical data of type'pandas.Series' taken from'pandas.DataFrame'.

    Returns
    ---------
    list. One-dimensional numerical list.

    Error
    ---------
        ValueError: List content must be one numerical data: chack_data=", chack_data.
        Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
        non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If 'chack_data' is 'True', it means that 'data' does not need to 
        be changed; if it is'False', the input 'data' is changed to a one-dimensional list of numerical data.

        ValueError: The type of one-dimensional data must be 'pandas.Series', and the content is numeric data.
        Solution: Input 'data' must be a one-dimensional numerical data of type'pandas.Series' taken from'pandas.DataFrame'.

    """
    try:
        type_data = str(type(data))
        chack_data = Math.chack_list_all_num(data)
        if(type_data == "<class 'pandas.core.series.Series'>"):
            if(chack_data):
                return np.array(data).tolist()
            else:
                raise ValueError("List content must be one numerical data: chack_data=", chack_data)
        else:
            raise ValueError
    except:
        raise ValueError("The type of one-dimensional data must be 'pandas.Series', and the content is numeric data.")

def split_data(data, ratio: float):
    """Split the data into two data sets according to the input ratio.

    This function splits one-dimensional numerical list into two data sets according to the split ratio. 
    The first data set is the data length of the split ratio, and the second data set is the data length of the split ratio.

    Parameters
    ---------
        data: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

        ratio: float.
        Limitation factor: 0 < ratio <1. The split ratio is a floating point number used to determine the respective lengths of the two data sets after splitting.

    Returns
    ---------
    Two one-dimensional data sets. The first data set is the data length of the split ratio, and the second data set is the data length of the split ratio.

    Error
    ---------
        TypeError: 'data' of type 'NoneType'.
        Solution: 'data' must be one-dimensional numerical list.

        ValueError: 'ratio' must be between 0 and 1, excluding 0 and 1.
        Solution: Enter a floating point number greater than 0 and less than 1.

        ValueError: There are less than two numerical data in the list.
        Solution: There are at least two numerical data in the list.

    """
    try:
        if(ratio <= 0 or ratio >= 1 ):
            raise ValueError("'ratio' must be between 0 and 1, excluding 0 and 1.")
        else:
            if(len(data) > 2):
                num = int(len(data)*ratio)
                train_data = data[:num]
                test_data = data[-(len(data) - num):]
                return train_data, test_data
            else:
                raise ValueError("There are at least two numerical data in the list.")
    except TypeError as err:
        raise TypeError(err)

def create_ar_data(data: pandas.Series, lags: int = 1) -> pandas.DataFrame:
    """Create an autoregressive data set.

    The autoregressive data set created to facilitate the calculation of autoregressive, its type is'pandas.DataFrame'.
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

    Parameters
    ---------
        data: pandas.Series.
        'data' is a one-dimensional numerical data of type 'pandas.Series' taken from 'pandas.DataFrame'.

        lags: int , default =  1.
        'lags' represents the number of lagging periods. The default value is 1, which means the data is 1 period behind.

    Returns
    ---------
    pandas.DataFrame. The autoregressive data set created to facilitate the calculation of autoregressive.

    Error
    ---------
        ValueError: The type of one-dimensional data must be 'pandas.Series', and the content is numeric data.
        Solution: Check whether the parameter matches the 'ValueError' description, or use the default value.

        ValueError: Check that the input list is a one-dimensional numerical data.
        Solution: 'data' is one-dimensional data and the type is'pandas.Series', but the content is not all numeric.

        ValueError: 'lags' value must be a positive integer, and the condition is greater than 1 and less than the length of the data.
        Solution: Check whether the parameter matches the 'ValueError' description, or use the default value.

        TypeError: object of type 'NoneType' has no len().
        Solution:  Please do not enter 'None'.Check that the type of one-dimensional data must be 'pandas.Series', and the content is numeric data.

        KeyError: The 'data' type is'pandas.DataFrame', please enter the correct key value.
        Solution: Use the correct key value to obtain data.

    """
    try:
        type_data = str(type(data))
        chack_data = Math.chack_list_all_num(data)
        if(type_data != "<class 'pandas.core.series.Series'>"):
            raise ValueError("The type of one-dimensional data must be 'pandas.Series', and the content is numeric data.")
        if (chack_data == False):
            raise ValueError("Check that the input list is a one-dimensional numerical data.")
        if(lags >= 1 and type(lags) == int and lags < len(data)):
            lags_label = []
            for i in range(1, lags+1):
                data_i = data.shift(i)
                if (i == 1):
                    data1 = pandas.concat([data_i, data], axis=1)
                else:
                    data1 = pandas.concat([data_i, data1], axis=1)
            for i in range(lags, 0, -1):
                lags_label.append(('t-'+str(i)))
            lags_label.append('t')
            data1.columns = lags_label
            data1 = data1[lags:]
            return data1
        else:
            raise ValueError("'lags' value must be a positive integer, and the condition is greater than 1 and less than the length of the data.")
    except ValueError as err:
        raise ValueError(err)
    except KeyError:
        raise KeyError("The'data' type is'pandas.DataFrame', please enter the correct key value.")

def n_order_difference_data(data: pandas.Series, periods: int = 1, log: bool = False) -> pandas.Series:
    """Sequence of numbers after difference operation.

    The difference between the current period and the lag period data is called the first-order difference. 
    The difference the data a second time is called the second-order difference.

    Parameters
    ---------
        data: pandas.Series.
        'data' is a one-dimensional numerical data of type 'pandas.Series' taken from 'pandas.DataFrame'.

        periods: int , default =  1.
        'periods' represents the number of lagging periods. The default value is 1, which means the data is 1 period behind.

        log: bool, default  = False
        Does the data need to be multiplied by the natural logarithm.

    Returns
    ---------
    pandas.Series. Return the difference data.

    Error
    ---------
        ValueError: The type of one-dimensional data must be 'pandas.Series', and the content is numeric data.
        Solution: Check whether the parameter matches the 'ValueError' description, or use the default value.

        ValueError: Check that the input list is a one-dimensional numerical data.
        Solution: 'data' is one-dimensional data and the type is'pandas.Series', but the content is not all numeric.

        ValueError: 'periods' value must be a positive integer, and the condition is greater than 1 and less than the length of the data.
        Solution: Check whether the parameter matches the 'ValueError' description, or use the default value.

        TypeError: object of type 'NoneType' has no len().
        Solution:  Please do not enter 'None'.Check that the type of one-dimensional data must be 'pandas.Series', and the content is numeric data.

        KeyError: The 'data' type is'pandas.DataFrame', please enter the correct key value.
        Solution: Use the correct key value to obtain data.

    References
    ---------
    Stationarity and differencing: https://otexts.com/fpp2/stationarity.html
    Time series analysis ppt: http://homepage.ntu.edu.tw/~sschen/Book/Slides/Ch2Basic.pdf

    """
    try:
        if(periods >= 1 and type(periods) == int and periods < len(data)):
            diff_data = []
            ar_data = create_ar_data(data, periods)
            if(log == True):
                data_t_log = Math.log(ar_data['t'])
                data_per_log = Math.log(ar_data['t-'+str(periods)])
                diff_data = Math.sub(data_t_log, data_per_log)
                diff_df = pandas.DataFrame(diff_data, index=ar_data['t'].index)
            elif(log == False):
                diff_data = Math.sub(ar_data['t'], ar_data['t-'+str(periods)])
                diff_df = pandas.DataFrame(diff_data, index = ar_data['t'].index)
            return diff_df[0]
        else:
            raise ValueError("'periods' value must be a positive integer, and the condition is greater than 1 and less than the length of the data.")
    except ValueError as err:
        raise ValueError(err)
