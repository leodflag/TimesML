#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2020 by leodflag

import unittest
import pandas as pd
import numpy as np
import Math.Statistics as Math

class MathTestCase(unittest.TestCase):
    def setUp(self):
        self.__listA = [5, 9, 6, 7, 2, 2]
        self.__listB = [3, 6, 7, 2, 9, 5]
        self.__listC = [3.122, 6.145, 7.126, 2.125, 9.562, 5.123]
        self.__list_one_str = [3, 6, 7, '2', 9, 5]
        self.__list_two_dim = [[3, 6], [7, 2, 9, 5]]
        self.__list_empty = []
        self.__None = None
        self.__Str = '5'

    def test_chack_list_all_num(self):
        result_T = True
        result_F = False
        expected_T = Math.chack_list_all_num(self.__listA)
        expected_F = Math.chack_list_all_num(self.__list_one_str)
        expected_F_2 = Math.chack_list_all_num(self.__list_two_dim)
        expected_F_str = Math.chack_list_all_num(self.__Str)
        self.assertEqual(result_T, expected_T)
        self.assertEqual(result_F, expected_F)
        self.assertEqual(result_F, expected_F_2)
        self.assertEqual(result_F, expected_F_str)
        self.assertRaises(ValueError, Math.chack_list_all_num, self.__list_empty)
        self.assertRaises(TypeError, Math.chack_list_all_num, self.__None)

    def test_plus(self):
        result = [8, 15, 13, 9, 11, 7]
        expected = Math.plus(self.__listA, self.__listB)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Math.plus, self.__listB, self.__list_two_dim)
        self.assertRaises(ValueError, Math.plus, self.__listB, self.__list_one_str)
        self.assertRaises(ValueError, Math.plus, self.__list_empty, self.__listA)
        self.assertRaises(ValueError, Math.plus, self.__Str, self.__listA)
        self.assertRaises(TypeError, Math.plus, self.__None, self.__listA)

    def test_sub(self):
        result = [2, 3, -1, 5, -7, -3]
        expected = Math.sub(self.__listA, self.__listB)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Math.sub, self.__listB, self.__list_two_dim)
        self.assertRaises(ValueError, Math.sub, self.__listB, self.__list_one_str)
        self.assertRaises(ValueError, Math.sub, self.__list_empty, self.__listA)
        self.assertRaises(ValueError, Math.sub, self.__Str, self.__listA)
        self.assertRaises(TypeError, Math.sub, self.__None, self.__listA)

    def test_mult(self):
        result = [15, 54, 42, 14, 18, 10]
        expected = Math.mult(self.__listA, self.__listB)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Math.mult, self.__listB, self.__list_two_dim)
        self.assertRaises(ValueError, Math.mult, self.__listB, self.__list_one_str)
        self.assertRaises(ValueError, Math.mult, self.__list_empty, self.__listA)
        self.assertRaises(ValueError, Math.mult, self.__Str, self.__listA)
        self.assertRaises(TypeError, Math.mult, self.__None, self.__listA)

    def test_div(self):
        result = [1.6667,1.5, 0.8571, 3.5, 0.2222, 0.4]
        expected = Math.div(self.__listA, self.__listB)
        expected_round = []
        for i in range(len(expected)):
            expected_round.append(round(expected[i], 4))
        self.assertEqual(result, expected_round)
        self.assertRaises(ValueError, Math.div, self.__listB, self.__list_two_dim)
        self.assertRaises(ValueError, Math.div, self.__listA, self.__list_one_str)
        self.assertRaises(ValueError, Math.div, self.__list_empty, self.__listA)
        self.assertRaises(ValueError, Math.div, self.__Str, self.__listA)
        self.assertRaises(TypeError, Math.div, self.__None, self.__listA)

    def test_log(self):
        result = [1.0986, 1.7918, 1.9459, 0.6931, 2.1972, 1.6094]
        expected = Math.log(self.__listB)
        expected_round = []
        for i in range(len(expected)):
            expected_round.append(round(expected[i], 4))
        self.assertEqual(result, expected_round)
        self.assertRaises(ValueError, Math.log,  self.__list_one_str)
        self.assertRaises(ValueError, Math.log,  self.__list_two_dim)
        self.assertRaises(ValueError, Math.log,  self.__list_empty)
        self.assertRaises(ValueError, Math.log, self.__Str)
        self.assertRaises(TypeError, Math.log, self.__None)

    def test_round_list(self):
        result = [3.12, 6.14, 7.13, 2.12, 9.56, 5.12]
        expected = Math.round_list(self.__listC, 2)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Math.round_list,  self.__list_one_str, 2)
        self.assertRaises(ValueError, Math.round_list,  self.__list_two_dim, 2)
        self.assertRaises(ValueError, Math.round_list,  self.__list_empty, 2)
        self.assertRaises(ValueError, Math.round_list, self.__Str, 2)
        self.assertRaises(TypeError, Math.round_list, self.__None, 2)

    def test_mean(self):
        result = 5.1667
        expected = round(Math.mean(self.__listA), 4)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Math.mean,  self.__list_one_str)
        self.assertRaises(ValueError, Math.mean,  self.__list_two_dim)
        self.assertRaises(ValueError, Math.mean,  self.__list_empty)
        self.assertRaises(ValueError, Math.mean, self.__Str)
        self.assertRaises(TypeError, Math.mean, self.__None)

    def test_variance(self):
        result = 7.7667
        expected = round(Math.variance(self.__listA), 4)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Math.variance,  self.__list_one_str)
        self.assertRaises(ValueError, Math.variance,  self.__list_two_dim)
        self.assertRaises(ValueError, Math.variance,  self.__list_empty)
        self.assertRaises(ValueError, Math.variance, self.__Str)
        self.assertRaises(TypeError, Math.variance, self.__None)

    def test_standard_deviation(self):
        result = 2.7869
        expected = round(Math.standard_deviation(self.__listA), 4)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Math.standard_deviation,  self.__list_one_str)
        self.assertRaises(ValueError, Math.standard_deviation,  self.__list_two_dim)
        self.assertRaises(ValueError, Math.standard_deviation,  self.__list_empty)
        self.assertRaises(ValueError, Math.standard_deviation, self.__Str)
        self.assertRaises(TypeError, Math.standard_deviation, self.__None)

    def test_covariance(self):
        result = -2.4667
        expected = round(Math.covariance(self.__listA, self.__listB), 4)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Math.covariance, self.__listB, self.__list_two_dim)
        self.assertRaises(ValueError, Math.covariance, self.__listB, self.__list_one_str)
        self.assertRaises(ValueError, Math.covariance, self.__list_empty, self.__listA)
        self.assertRaises(ValueError, Math.covariance, self.__Str, self.__listA)
        self.assertRaises(TypeError, Math.covariance, self.__None, self.__listA)

    def test_coefficient_of_variation(self):
        result = 53.9395
        expected = round(Math.coefficient_of_variation(self.__listA), 4)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Math.coefficient_of_variation,  self.__list_one_str)
        self.assertRaises(ValueError, Math.coefficient_of_variation,  self.__list_two_dim)
        self.assertRaises(ValueError, Math.coefficient_of_variation,  self.__list_empty)
        self.assertRaises(ValueError, Math.coefficient_of_variation, self.__Str)
        self.assertRaises(TypeError, Math.coefficient_of_variation, self.__None)

    def test_correlation_coefficient(self):
        result = -0.3428
        expected = round(Math.correlation_coefficient(self.__listA, self.__listB), 4)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Math.correlation_coefficient, self.__listB, self.__list_two_dim)
        self.assertRaises(ValueError, Math.correlation_coefficient, self.__listB, self.__list_one_str)
        self.assertRaises(ValueError, Math.correlation_coefficient, self.__list_empty, self.__listA)
        self.assertRaises(ValueError, Math.correlation_coefficient, self.__Str, self.__listA)
        self.assertRaises(TypeError, Math.correlation_coefficient, self.__None, self.__listA)

    def test_coefficient_of_determination(self):
        train = [2,2,3,5,6,7]
        test = [6, 7 ]
        predict = [5, 5]
        result = 0.5610
        expected = round(Math.coefficient_of_determination(train, test, predict), 4)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Math.coefficient_of_determination, train, self.__listB, self.__list_two_dim)
        self.assertRaises(ValueError, Math.coefficient_of_determination, train, self.__listB, self.__list_one_str)
        self.assertRaises(ValueError, Math.coefficient_of_determination, train, self.__list_empty, self.__listA)
        self.assertRaises(ValueError, Math.coefficient_of_determination, train, self.__Str, self.__listA)
        self.assertRaises(TypeError, Math.coefficient_of_determination, train, self.__None, self.__listA)

    def test_mean_square_error(self):
        result = 16.1667
        expected = round(Math.mean_square_error(self.__listA, self.__listB), 4)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Math.mean_square_error, self.__listB, self.__list_two_dim)
        self.assertRaises(ValueError, Math.mean_square_error, self.__listB, self.__list_one_str)
        self.assertRaises(ValueError, Math.mean_square_error, self.__list_empty, self.__listA)
        self.assertRaises(ValueError, Math.mean_square_error, self.__Str, self.__listA)
        self.assertRaises(TypeError, Math.mean_square_error, self.__None, self.__listA)

    def test_mean_absolute_error(self):
        result = 3.5
        expected = round(Math.mean_absolute_error(self.__listA, self.__listB), 4)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Math.mean_absolute_error, self.__listB, self.__list_two_dim)
        self.assertRaises(ValueError, Math.mean_absolute_error, self.__listB, self.__list_one_str)
        self.assertRaises(ValueError, Math.mean_absolute_error, self.__list_empty, self.__listA)
        self.assertRaises(ValueError, Math.mean_absolute_error, self.__Str, self.__listA)
        self.assertRaises(TypeError, Math.mean_absolute_error, self.__None, self.__listA)

    def test_root_mean_square_error(self):
        result = 4.0208
        expected = round(Math.root_mean_squard_error(self.__listA, self.__listB), 4)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Math.root_mean_squard_error, self.__listB, self.__list_two_dim)
        self.assertRaises(ValueError, Math.root_mean_squard_error, self.__listB, self.__list_one_str)
        self.assertRaises(ValueError, Math.root_mean_squard_error, self.__list_empty, self.__listA)
        self.assertRaises(ValueError, Math.root_mean_squard_error, self.__Str, self.__listA)
        self.assertRaises(TypeError, Math.root_mean_squard_error, self.__None, self.__listA)

    def test_normalized_mean_square_error(self):
        result = 0.5744
        expected = round(Math.normalized_mean_squard_error(self.__listA, self.__listB), 4)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Math.normalized_mean_squard_error, self.__listB, self.__list_two_dim)
        self.assertRaises(ValueError, Math.normalized_mean_squard_error, self.__listB, self.__list_one_str)
        self.assertRaises(ValueError, Math.normalized_mean_squard_error, self.__list_empty, self.__listA)
        self.assertRaises(ValueError, Math.normalized_mean_squard_error, self.__Str, self.__listA)
        self.assertRaises(TypeError, Math.normalized_mean_squard_error, self.__None, self.__listA)

    def test_ACF(self):
        result = -0.3428
        expected = round(Math.ACF(self.__listA, self.__listB), 4)
        self.assertEqual(result, expected)
        self.assertRaises(ValueError, Math.ACF, self.__listB, self.__list_two_dim)
        self.assertRaises(ValueError, Math.ACF, self.__listB, self.__list_one_str)
        self.assertRaises(ValueError, Math.ACF, self.__list_empty, self.__listA)
        self.assertRaises(ValueError, Math.ACF, self.__Str, self.__listA)
        self.assertRaises(TypeError, Math.ACF, self.__None, self.__listA)

if __name__ == "__main__":
    unittest.main()
