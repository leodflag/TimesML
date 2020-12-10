#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2020 by leodflag

import unittest
import pandas as pd
import numpy as np
import TimeSeriesAnalysis.ProcessData as Data
import TimeSeriesAnalysis.Model as model

class TimeSeriesDataTestCase(unittest.TestCase):
    def setUp(self):
        self.__split_ratio = 0.7
        self.__lags = 3
        df_data = pd.read_csv('test_data/TWD_unittest.csv', index_col='date')
        self.__data = df_data['close']
        train, test = Data.split_data(self.__data, self.__split_ratio)
        self.__ar_model = model.AutoRegressive(3)
        self.__ma_model = model.MovingAverage(3)
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
        self.__test_ma_data = pd.DataFrame({
            "t-2": [0.21, -0.02, 0.04],
            "t-1": [-0.02, 0.04, -0.05],
            "t": [0.04, -0.05, 0.13],
            "Xt": [28.87, 28.92, 28.79]},
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
        expected = Data.list_to_dataframe(self.__list_one_dim_data)
        self.assertEqual(df_result[0].tolist(), expected[0].tolist())
        self.assertEqual(df_result.index.tolist(), expected.index.tolist())
        self.assertRaises(ValueError, Data.list_to_dataframe, self.__test_ar_data)
        self.assertRaises(ValueError, Data.list_to_dataframe, self.__list_empty)
        self.assertRaises(ValueError, Data.list_to_dataframe, self.__Str)
        self.assertRaises(ValueError, Data.list_to_dataframe, self.__None)
        self.assertRaises(ValueError, Data.list_to_dataframe, self.__list_two_dim_data)

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

    def test_create_ma_data(self):
        result = Data.create_ma_data(self.__test, 2)
        self.assertEqual(round(self.__test_ma_data['t-2'][0], 2), round(result['t-2'][0], 2))
        self.assertRaises(ValueError, Data.create_ma_data, self.__test, 1.2)
        self.assertRaises(ValueError, Data.create_ma_data, self.__test_ma_data['t'], 10)
        self.assertRaises(ValueError, Data.create_ma_data, self.__list_one_dim_data, 2)
        self.assertRaises(ValueError, Data.create_ma_data, self.__list_two_dim_data, 2)
        self.assertRaises(ValueError, Data.create_ma_data, self.__list_empty, 2)
        self.assertRaises(ValueError, Data.create_ma_data, self.__Str, 2)
        self.assertRaises(TypeError, Data.create_ma_data, self.__None, 2)
        self.assertRaises(KeyError, Data.create_ma_data, self.__test_ma_data)

    def test_cal_deviation(self):
        self.__ar_model.fit(self.__test)
        result = -0.13
        self.assertEqual(round(self.__ar_model.deviation, 2), result)

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
        result = [29.013252216901748, 29.008074480236388, 29.059593367035852]
        self.__ar_model.fit(self.__train)
        self.__ar_model.predict(self.__test)
        self.assertEqual(result, self.__ar_model.prediction)

    def test_cal_ma_normal_equation(self):
        result = np.array(
            [[29.10611059319247],
            [-1.1292229645448373],
            [-0.21150889116160343],
            [0.05024933178532365]]
        )
        self.__ma_model.fit(self.__train)
        self.assertEqual(result.tolist(), self.__ma_model.coefficients_matrix.tolist())

    def test_ma_predict(self):
        result = [29.025724212108823, 29.283106399137544]
        self.__ma_model.fit(self.__train)
        self.__ma_model.predict(self.__test)
        self.assertEqual(result, self.__ma_model.prediction)


if __name__ == "__main__":
    unittest.main()
