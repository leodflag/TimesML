#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2020 by leodflag

import unittest
import pandas as pd
import numpy as np
import Math.Statistics as Math
import TimeSeriesAnalysis.ProcessData as Data
import TimeSeriesAnalysis.Model as model

class TimeSeriesDataTestCase(unittest.TestCase):

    """

    Unit test for processing time series data

    """
    def setUp(self):  # 每個測試運行前都會執行
        self.__split_ratio = 0.7
        self.__lags = 3
        df_data = pd.read_csv('test_data/TWD_unittest.csv', index_col='date')
        self.__data = df_data['close']
        train, test = Data.split_data(self.__data, self.__split_ratio)
        self.__ar_model = model.AutoRegressive(3)
        self.__sma_model = model.SimpleMovingAverage(2)
        self.__test = test
        self.__train = train
        self.__list_one_dim_data = [5, 9, 6, 7, 2, 3]
        self.__list_two_dim_data = [[1, 2, 3], [8, 7]]
        self.__list_empty = []
        self.__None = None
        self.__Str = '5'
        self.__test_ar_data = pd.DataFrame({
            "t-3": [29.10, 28.89, 28.91],
            "t-2": [28.89, 28.91, 28.87],
            "t-1": [28.91, 28.87, 28.92],
            "t": [28.87, 28.92, 28.79]},
            index=['2020-10-01 23:00:00', '2020-10-04 23:00:00', '2020-10-06 13:56:21'])

    def test_create_sequence_list(self):
        result = [1, 2, 3, 4, 5, 6]
        expected = Data.create_sequence_list(self.__list_one_dim_data)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Data.create_sequence_list, self.__test_ar_data)
        self.assertRaises(ValueError, Data.create_sequence_list, self.__list_two_dim_data)
        self.assertRaises(ValueError, Data.create_sequence_list, self.__Str)
        self.assertRaises(ValueError, Data.create_sequence_list, self.__list_empty)
        self.assertRaises(ValueError, Data.create_sequence_list, self.__None)

    def test_list_to_dataframe(self):
        result_index = [1, 2, 3, 4, 5, 6]
        df_result = pd.DataFrame(self.__list_one_dim_data, index=result_index)
        expected = Data.list_to_dataframe(self.__list_one_dim_data, result_index)
        expected2 = Data.list_to_dataframe(self.__test, self.__test.index)
        self.assertEqual(df_result[0].tolist(), expected[0].tolist())
        self.assertEqual(df_result.index.tolist(), expected.index.tolist())
        self.assertEqual(self.__test.tolist() , expected2['close'].tolist())
        self.assertRaises(ValueError, Data.list_to_dataframe, self.__test_ar_data, self.__list_one_dim_data)
        self.assertRaises(ValueError, Data.list_to_dataframe, self.__list_empty, self.__list_one_dim_data)
        self.assertRaises(ValueError, Data.list_to_dataframe, self.__Str, self.__list_one_dim_data)
        self.assertRaises(ValueError, Data.list_to_dataframe, self.__None, self.__list_one_dim_data)
        self.assertRaises(ValueError, Data.list_to_dataframe, self.__list_two_dim_data, self.__list_one_dim_data)

    def test_dataframe_to_list(self):
        result = [28.87, 28.92, 28.79]
        expected = Data.dataframe_to_list(self.__test_ar_data['t'])
        df_one_str = pd.DataFrame({'t': [28.87, '28.92', 28.79]}, index = [0, 1, 2])
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Data.dataframe_to_list, df_one_str['t'])
        self.assertRaises(ValueError, Data.dataframe_to_list, self.__test_ar_data)
        self.assertRaises(ValueError, Data.dataframe_to_list, self.__list_empty)
        self.assertRaises(ValueError, Data.dataframe_to_list, self.__Str)
        self.assertRaises(ValueError, Data.dataframe_to_list, self.__None)
        self.assertRaises(ValueError, Data.dataframe_to_list, self.__list_two_dim_data)

    def test_split_data(self):
        data = []
        train_data, test_data = Data.split_data(self.__data, self.__split_ratio)
        self.assertEqual(len(train_data), 12)
        self.assertEqual(len(test_data), 6)
        self.assertRaises((TypeError, ValueError), Data.split_data, data,0 )
        self.assertRaises((TypeError, ValueError), Data.split_data, data, 0.2)

    def test_create_ar_data(self):
        result = Data.create_ar_data(self.__test, 3)
        self.assertEqual(self.__test_ar_data['t-2'].tolist(), result['t-2'].tolist())
        self.assertRaises(ValueError, Data.create_ar_data, self.__test, 2.2)
        self.assertRaises(ValueError, Data.create_ar_data, self.__test_ar_data['t'], 10)
        self.assertRaises(ValueError, Data.create_ar_data, self.__list_one_dim_data, 2)
        self.assertRaises(ValueError, Data.create_ar_data, self.__list_two_dim_data, 2)
        self.assertRaises(ValueError, Data.create_ar_data, self.__list_empty, 2)
        self.assertRaises(ValueError, Data.create_ar_data, self.__Str, 2)
        self.assertRaises(TypeError, Data.create_ar_data, self.__None, 2)
        self.assertRaises(KeyError, Data.create_ar_data, self.__test_ar_data)

    def test_n_order_difference_data(self):
        result = [-0.21, 0.02, -0.04, 0.05, -0.13]
        result_log = [-0.007243,  0.000692, -0.001385, 0.001730, -0.004505]
        expected = Data.n_order_difference_data(self.__test)
        expected_list = Data.dataframe_to_list(expected)
        expected_list = Math.round_list(expected_list, 2)
        expected_log = Data.n_order_difference_data(self.__test, 1, True)
        expected_log_list = Data.dataframe_to_list(expected_log)
        expected_log_list = Math.round_list(expected_log_list, 6)
        self.assertEqual(result, expected_list)
        self.assertEqual(result_log, expected_log_list)
        self.assertRaises(ValueError, Data.n_order_difference_data, self.__test, 1.2)
        self.assertRaises(ValueError, Data.n_order_difference_data, self.__test_ar_data['t'], 10)
        self.assertRaises(ValueError, Data.n_order_difference_data, self.__list_one_dim_data, 2)
        self.assertRaises(ValueError, Data.n_order_difference_data, self.__list_two_dim_data, 2)
        self.assertRaises(ValueError, Data.n_order_difference_data, self.__list_empty, 2)
        self.assertRaises(ValueError, Data.n_order_difference_data, self.__Str, 2)
        self.assertRaises(TypeError, Data.n_order_difference_data, self.__None, 2)
        self.assertRaises(KeyError, Data.n_order_difference_data, self.__test_ar_data)

    def test_cal_ar_normal_equation(self):
        result_A = np.array(
            [[192.0],
             [-3.0],
             [-1.0],
             [0.28125]]
        )
        self.__ar_model.fit(self.__test)
        self.assertEqual(result_A.tolist(), self.__ar_model.coefficients_matrix.tolist())

    def test_ar_predict(self):
        result = [29.023252216901746, 29.15093198716034, 29.225612724895058]
        self.__ar_model.fit(self.__train)
        self.__ar_model.predict(self.__test, True)
        self.assertEqual(result, self.__ar_model.test_predict.tolist())

    def test_SimpleMovingAverage_sma_result(self):
        result = [28.995, 28.900, 28.890, 28.895, 28.855]
        self.__sma_model.fit(self.__test)
        expected = Math.round_list(self.__sma_model.sma_result, 3)
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
