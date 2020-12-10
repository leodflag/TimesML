#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2020 by leodflag

import unittest
import pandas as pd
import numpy as np
import TimeSeriesAnalysis.ProcessData as Data
import TimeSeriesAnalysis.Model as model

class ModelTestCase(unittest.TestCase):
    def setUp(self):
        self.__split_ratio = 0.7
        self.__lags = 3
        df_data = pd.read_csv('test_data/TWD_unittest.csv', index_col='date')
        self.__data = df_data['close']
        train, test = Data.split_data(self.__data, self.__split_ratio)
        self.__test = test
        self.__list_one_dim_data = [5, 9, 6, 7, 2, 3]
        self.__list_two_dim_data = [[1, 2, 3], [8, 7]]
        self.__test_ar_data = pd.DataFrame({
            "t-3": [29.10, 28.89, 28.91],
            "t-2": [28.89, 28.91, 28.87],
            "t-1": [28.91, 28.87, 28.92],
            "t": [28.87, 28.92, 28.79]},
            index=['2020-10-01 23:00:00', '2020-10-04 23:00:00', '2020-10-06 13:56:21'])
        self.__test_ma_data = pd.DataFrame({
            "t-2": [0.21, -0.02, 0.04],
            "t-1": [-0.02, 0.04, -0.05],
            "t": [0.04, -0.05, 0.13],
            "Xt": [28.87, 28.92, 28.79]},
            index=['2020-10-01 23:00:00', '2020-10-04 23:00:00', '2020-10-06 13:56:21'])

