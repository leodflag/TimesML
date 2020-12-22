#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2020 by leodflag

"""This module contains statistics calculation function.

Addition, subtraction, multiplication, and division of lists, as well as basic statistical calculation functions 
such as variance, standard deviation, and autocorrelation coefficient.

Functions
-----------
chack_list_all_num(listA) -> bool
	Determine the content of the list as one numerical data.

plus(listA, listB) -> list
	The two lists are added element by element.

sub(listA, listB) -> list
	Subtract two lists element by element.

mult(listA, listB) -> list
	Multiply two lists element by element.

div(listA, listB) -> list
	Divide two lists element by element.

log(listA) -> list:
    Multiply all the numerical elements in the list by the natural logarithm.

round_list(listA, ndigits: int) -> list:
    Take all the values in the list to a few digits after the decimal point.

mean(listA) -> float
	Calculate arithmetic mean of numerical data in a list.

variance(listA) -> float
	Calculate the variance of all elements in a list.

standard_deviation(listA) -> float
	Calculate the standard deviation of all elements in a list.

covariance(listA, listB) -> float
	Calculate the covariance of two lists of one-dimensional data.

coefficient_of_variation(listA) -> float
	Calculate the coefficient of variation of all elements in a list.

correlation_coefficient(listA, listB) -> float
	Calculate the linear correlation coefficient  of two lists of one-dimensional data.

coefficient_of_determination(listA, listB) -> float
	Calculate the coefficient of determination  of two lists of one-dimensional data.

mean_square_error(target_list, predict_list) -> float
	The regression index used to calculate the forecast error. Calculate the mean square error  of two lists of one-dimensional data.

mean_absolute_error(target_list, predict_list) -> float:
    The regression index used to calculate the forecast error. Calculate the mean absolute error(MAE) of two lists of one-dimensional data.

root_mean_squard_error(target_list, predict_list) -> float:
    The regression index used to calculate the forecast error. Calculate the root mean square error(RMSE) of two lists of one-dimensional data.

normalized_mean_squard_error(target_list, predict_list) -> float:
    The regression index used to calculate the forecast error. Calculate the normalized mean square error(NRMSE) of two lists of one-dimensional data.

ACF(listA, listB) -> float
	Calculate the autocorrelation coefficients of two lists of one-dimensional data.

Notes
-----------
Pay attention to the input data type of the function.

Use the function "list_to_dataframe" or "dataframe_to_list" of the module "TimeSeriesAnalysis.ProcessData" to convert the data type.

"""
import math
import numpy as np
import pandas as pd

def chack_list_all_num(listA) -> bool:
    """Determine the content of the list as one numerical data.

    This function is to check the input list, the content is a one-dimensional numerical list.
    If the input is not a one-dimensional list, or the content of the one-dimensional list is not all numerical data, 
    it will be returned 'False', otherwise it will return 'True'.

    Parameters
    ---------
        listA : list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    bool(True or False).

    'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, 
    and one-dimensional non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5].

    Error
    ---------
        ValueError: The list must be one-dimensional numerical data, and there is at least one numerical data in it.
        Solution: The entered'listA' is an empty list. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

        TypeError: object of type 'NoneType' has no len()
        Solution:  Please do not enter 'None'.Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    """
    try:
        if(len(listA) != 0):
            for i in range(len(listA)):
                if(type(listA[i]) == int or type(listA[i]) == float or type(listA[i]) == np.float64):
                    continue
                else:
                    return False
            return True
        else:
            raise ValueError("The list must be one-dimensional numerical data, and there is at least one numerical data in it.")
    except KeyError:
        raise KeyError("The'data' type is'pandas.DataFrame', please enter the correct key value.")
    except TypeError as err:
        raise TypeError(err)


def plus(listA, listB) -> list:
    """The two lists are added element by element.

    Addition: The two lists are added element by element, and return a list of sums.

    Parameters
    ---------
        listA: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

        listB: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    list. The result of adding the two lists.

    Error
    ---------
        ValueError: The two lists must be the same length.
        Solution: Make sure that the number of numerical data in the two lists is the same (the same length).

        ValueError: List content must be one-dimensional numerical data: listA=",chack_A, "; listB=", chack_B.
        Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
        non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_A' is'True', it means that 'listA' does not need to be 
        changed; if it is'False', the input'listA' is changed to a one-dimensional list of numerical data.ListB has the same judgment and processing 
        method as listA.

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' or 'listB' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    """
    try:
        chack_A = chack_list_all_num(listA)
        chack_B = chack_list_all_num(listB)
        if(chack_A and chack_B):
            if (len(listA) == len(listB)):
                result_list = []
                for i in range(len(listA)):
                    result_list.append(listA[i] + listB[i])
                return result_list
            else:
                raise ValueError("The two lists must be the same length.")
        else:
            raise ValueError("List content must be one-dimensional numerical data: listA=",chack_A, "; listB=", chack_B)
    except TypeError as err:
        raise TypeError(err)

def sub(listA, listB) -> list:
    """Subtract two lists element by element.

    Subtraction: Subtract the elements of two lists pair by pair, and return a list of results.

    Parameters
    ---------
        listA: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

        listB: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    list. The result of subtracting the two lists.

    Error
    ---------
        ValueError: The two lists must be the same length.
        Solution: Make sure that the number of numerical data in the two lists is the same (the same length).

        ValueError: List content must be one-dimensional numerical data: listA=",chack_A, "; listB=", chack_B.
        Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
        non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_A' is'True', it means that 'listA' does not need to be 
        changed; if it is'False', the input 'listA' is changed to a one-dimensional list of numerical data.ListB has the same judgment and processing 
        method as listA.

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' or 'listB' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    """
    try:
        chack_A = chack_list_all_num(listA)
        chack_B = chack_list_all_num(listB)
        if(chack_A and chack_B):
            if (len(listA) == len(listB)):
                result_list = []
                for i in range(len(listA)):
                    result_list.append(listA[i] - listB[i])
                return result_list
            else:
                raise ValueError("The two lists must be the same length.")
        else:
            raise ValueError("List content must be one numerical data: listA=",chack_A, "; listB=", chack_B)
    except TypeError as err:
        raise TypeError(err)

def mult(listA, listB) -> list:
    """Multiply two lists element by element.

    Multiplication: The elements of two lists are multiplied pair by pair to return a list of results.

    Parameters
    ---------
        listA: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

        listB: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    list. The result of multiplication.

    Error
    ---------
        ValueError: The two lists must be the same length.
        Solution: Make sure that the number of numerical data in the two lists is the same (the same length).

        ValueError: List content must be one-dimensional numerical data: listA=",chack_A, "; listB=", chack_B.
        Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
        non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_A' is'True', it means that 'listA' does not need to be 
        changed; if it is'False', the input 'listA' is changed to a one-dimensional list of numerical data.ListB has the same judgment and processing 
        method as listA.

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' or 'listB' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    """
    try:
        chack_A = chack_list_all_num(listA)
        chack_B = chack_list_all_num(listB)
        if(chack_A and chack_B):
            if (len(listA) == len(listB)):
                result_list = []
                for i in range(len(listA)):
                    result_list.append(listA[i] * listB[i])
                return result_list
            else:
                raise ValueError("The two lists must be the same length.")
        else:
            raise ValueError("List content must be one numerical data: listA=",chack_A, "; listB=", chack_B)
    except TypeError as err:
        raise TypeError(err)

def div(listA, listB) -> list:
    """Divide two lists element by element.

    Division: Divide the elements of two lists pair by pair, and return a list of results.

    Parameters
    ---------
        listA: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

        listB: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    list. Result of division.

    Error
    ---------
        ValueError: The two lists must be the same length.
        Solution: Make sure that the number of numerical data in the two lists is the same (the same length).

        ValueError: List content must be one-dimensional numerical data: listA=",chack_A, "; listB=", chack_B.
        Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
        non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_A' is'True', it means that 'listA' does not need to be 
        changed; if it is'False', the input 'listA' is changed to a one-dimensional list of numerical data.ListB has the same judgment and processing 
        method as listA.

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' or 'listB' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    """
    try:
        chack_A = chack_list_all_num(listA)
        chack_B = chack_list_all_num(listB)
        if(chack_A and chack_B):
            if (len(listA) == len(listB)):
                result_list = []
                for i in range(len(listA)):
                    result_list.append(listA[i] / listB[i])
                return result_list
            else:
                raise ValueError("The two lists must be the same length.")
        else:
            raise ValueError("List content must be one numerical data: listA=",chack_A, "; listB=", chack_B)
    except TypeError as err:
        raise TypeError(err)

def log(listA) -> list:
    """Multiply all the numerical elements in the list by the natural logarithm.

    Parameters
    ---------
        listA: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    list. Result of natural logarithm list.

    Error
    ---------
        ValueError: List content must be one-dimensional numerical data: listA=",chack_A".
        Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
        non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_A' is'True', it means that 'listA' does not need to be 
        changed; if it is'False', the input 'listA' is changed to a one-dimensional list of numerical data.

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' or 'listB' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    References
    ---------
    log: https://en.wikipedia.org/wiki/Natural_logarithm

    """
    try:
        chack_A = chack_list_all_num(listA)
        if(chack_A):
            result_list = []
            for i in range(len(listA)):
                if (listA[i] > 0):
                    result_list.append(math.log(listA[i]))
                else:
                    result_list.append(0)
            return result_list
        else:
            raise ValueError("List content must be one numerical data: listA=",chack_A)
    except TypeError as err:
        raise TypeError(err)


def round_list(listA, ndigits: int) -> list:
    """Take all the values in the list to a few digits after the decimal point.

    Parameters
    ---------
        listA : list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

        ndigits: int
        A few digits after the decimal point.

    Returns
    ---------
    list. Return a list of values with a few digits after the decimal point.

    Error
    ---------
        ValueError: The list must be one-dimensional numerical data, and there is at least one numerical data in it.
        Solution: The entered'listA' is an empty list. Please change to a one-dimensional list of numerical data.

        ValueError: The list must be one-dimensional numerical data.
        Solution: 'listA' contains a string or is two-dimensional list. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    """
    try:
        chack_A = chack_list_all_num(listA)
        if(chack_A):
            result_list = []
            for i in range(len(listA)):
                result_list.append(round(listA[i], ndigits))
            return result_list
        else:
            raise ValueError("List content must be one numerical data: listA=",chack_A)
    except TypeError as err:
        raise TypeError(err)

def mean(listA) -> float:
    """Calculate arithmetic mean of numerical data in a list.

    Mean: Add all the numerical data in the list and divide by the length of the list.

    Parameters
    ---------
        listA : list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    float. Arithmetic mean of numerical data in a list.

    Error
    ---------
        ValueError: The list must be one-dimensional numerical data, and there is at least one numerical data in it.
        Solution: The entered'listA' is an empty list. Please change to a one-dimensional list of numerical data.

        ValueError: The list must be one-dimensional numerical data.
        Solution: 'listA' contains a string or is two-dimensional list. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    References
    ---------
    Mean: https://zh.wikipedia.org/wiki/%E5%B9%B3%E5%9D%87%E6%95%B0

    """
    try:
        chack_A = chack_list_all_num(listA)
        if (chack_A):
            Sum = 0
            for i in range(len(listA)):
                Sum += listA[i]
            return Sum/(len(listA))
        else:
            raise ValueError("The list must be one-dimensional numerical data.")
    except TypeError as err:
        raise TypeError(err)

def variance(listA) -> float:
    """Calculate the variance of all elements in a list.

    Variance:The average of the squared distance of each number from its mean. 
    Algorithm: All the numerical data in the list are subtracted one by one from the mean value 
    and then squared, after adding them up and dividing by the length of the list minus one.

    Parameters
    ---------
        listA : list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    float. The variance of all elements in a list.

    Error
    ---------
        ValueError: The list must be one-dimensional numerical data, and there is at least one numerical data in it.
        Solution: The entered'listA' is an empty list. Please change to a one-dimensional list of numerical data.

        ValueError: The list must be one-dimensional numerical data.
        Solution: 'listA' contains a string or is two-dimensional list. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    References
    ---------
    Variance: https://zh.wikipedia.org/wiki/%E6%96%B9%E5%B7%AE

    """
    try:
        listA_mean = mean(listA)
        Sum = 0
        for i in range(len(listA)):
            num = 0
            num = (listA[i] - listA_mean)**2
            Sum += num
        return Sum / (len(listA) - 1)
    except TypeError as err:
        raise TypeError(err)

def standard_deviation(listA) -> float:
    """Calculate the standard deviation of all elements in a list.

    Standard deviation: Square root of variance. All the numerical data in the list are subtracted one by one 
    from the mean value and then squared. Algorithm: After adding them up, divide by the length of the list minus 
    one, and finally take the square root.

    Parameters
    ---------
        listA: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    float. The standard deviation of all elements in a list.

    Error
    ---------
        ValueError: The list must be one-dimensional numerical data, and there is at least one numerical data in it.
        Solution: The entered'listA' is an empty list. Please change to a one-dimensional list of numerical data.

        ValueError: The list must be one-dimensional numerical data.
        Solution: 'listA' contains a string or is two-dimensional list. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    References
    ---------
    Standard deviation: https://zh.wikipedia.org/wiki/%E6%A8%99%E6%BA%96%E5%B7%AE

    """
    try:
        listA_mean = mean(listA)
        Sum = 0
        for i in range(len(listA)):
            num = 0
            num = (listA[i] - listA_mean)**2
            Sum += num
        return (Sum /( len(listA) - 1))**0.5
    except TypeError as err:
        raise TypeError(err)

def covariance(listA, listB) -> float:
    """Calculate the covariance of two lists of one-dimensional data.

    Covariance: After calculating the respective averages of the two lists, subtract the averages one by one and 
    multiply them to form a value. Algorithm: After the entire list is calculated, all the values are added up, and finally
    divided by the length of the list minus one.

    Parameters
    ---------
        listA: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

        listB: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    float. The covariance of two lists of one-dimensional data.

    Error
    ---------
        ValueError: The two lists must be the same length.
        Solution: Make sure that the number of numerical data in the two lists is the same (the same length).

        ValueError: List content must be one-dimensional numerical data: listA=",chack_A, "; listB=", chack_B.
        Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
        non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_A' is'True', it means that 'listA' does not need to be 
        changed; if it is'False', the input 'listA' is changed to a one-dimensional list of numerical data.ListB has the same judgment and processing 
        method as listA.

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' or 'listB' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    References
    ---------
    Covariance: https://zh.wikipedia.org/wiki/%E5%8D%8F%E6%96%B9%E5%B7%AE

    """
    try:
        listA_mean = mean(listA)
        listB_mean = mean(listB)
        length = len(listA)
        Sum = 0
        for i in range(length):
            Sum += (listA[i] - listA_mean)*(listB[i] - listB_mean)
        return Sum / (length - 1)
    except TypeError as err:
        raise TypeError(err)

def coefficient_of_variation(listA) -> float:
    """Calculate the coefficient of variation of all elements in a list.

    Coefficient of variation: Can compare the standard deviation between different unit variables. 
    Algorithm: First calculate the mean and standard deviation of the list, and finally divide the standard deviation 
    by the mean and multiply by 100.

    Parameters
    ---------
        listA : list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    float. The coefficient of variation of all elements in a list.

    Error
    ---------
        ValueError: The list must be one-dimensional numerical data, and there is at least one numerical data in it.
        Solution: The entered'listA' is an empty list. Please change to a one-dimensional list of numerical data.

        ValueError: The list must be one-dimensional numerical data.
        Solution: 'listA' contains a string or is two-dimensional list. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    References
    ---------
    Coefficient of variation: https://zh.wikipedia.org/wiki/%E5%8F%98%E5%BC%82%E7%B3%BB%E6%95%B0

    """
    try:
        listA_mean = mean(listA)
        listA_sd = standard_deviation(listA)
        return (listA_sd / listA_mean)*100
    except TypeError as err:
        raise TypeError(err)

def correlation_coefficient(listA, listB) -> float:
    """Calculate the linear correlation coefficient  of two lists of one-dimensional data.

    Correlation coefficient: The strength of linear correlation between two variables. 
    Algorithm: Calculate the linear correlation coefficient according to the formula.

    Parameters
    ---------
        listA : list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

        listB : list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    float. The linear correlation coefficient  of two lists of one-dimensional data.

    Error
    ---------
        ValueError: The two lists must be the same length.
        Solution: Make sure that the number of numerical data in the two lists is the same (the same length).

        ValueError: List content must be one-dimensional numerical data: listA=",chack_A, "; listB=", chack_B.
        Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
        non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_A' is'True', it means that 'listA' does not need to be 
        changed; if it is'False', the input 'listA' is changed to a one-dimensional list of numerical data.ListB has the same judgment and processing 
        method as listA.

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' or 'listB' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    References
    ---------
    Correlation coefficient: https://www.statisticshowto.com/probability-and-statistics/correlation-coefficient-formula/

    """
    try:
        listAB_mult_sum = sum(mult(listA, listB))
        listAB_sum_mult = sum(listA) * sum(listB)
        listAA_mult_sum = sum(mult(listA, listA))
        listAA_sum_mult = sum(listA) * sum(listA)
        listBB_mult_sum = sum(mult(listB, listB))
        listBB_sum_mult = sum(listB) * sum(listB)
        n = len(listA)
        r =( (n*listAB_mult_sum - listAB_sum_mult) /
            (( (n*listAA_mult_sum - listAA_sum_mult)*(n*listBB_mult_sum - listBB_sum_mult) )**0.5) )
        return r
    except TypeError as err:
        raise TypeError(err)

def coefficient_of_determination(listA, listB) -> float:
    """Calculate the coefficient of determination  of two lists of one-dimensional data.

    Coefficient of determination: Explainable variation / total variation. A measure of how much proportional 
    dependent variable variation can be explained by regression lines and independent variables. The closer to 1, 
    the variation explained by this regression line is close to 100%. Algorithm: Square after calculating the linear correlation coefficient.

    Parameters
    ---------
        listA : list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

        listB : list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    float. The coefficient of determination  of two lists of one-dimensional data.

    Error
    ---------
        ValueError: The two lists must be the same length.
        Solution: Make sure that the number of numerical data in the two lists is the same (the same length).

        ValueError: List content must be one-dimensional numerical data: listA=",chack_A, "; listB=", chack_B.
        Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
        non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_A' is'True', it means that 'listA' does not need to be 
        changed; if it is'False', the input 'listA' is changed to a one-dimensional list of numerical data.ListB has the same judgment and processing 
        method as listA.

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' or 'listB' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    References
    ---------
    Coefficient of determination: https://zh.wikipedia.org/wiki/%E5%86%B3%E5%AE%9A%E7%B3%BB%E6%95%B0

    """
    try:
        R = correlation_coefficient(listA, listB)
        return R**2
    except TypeError as err:
        raise TypeError(err)

def mean_square_error(target_list, predict_list) -> float:
    """Regression indicators for numerical prediction. Calculate the mean square error (MSE) of two lists of one-dimensional data.

    Mean square error: Measures the mean of the square of the difference between the predicted value and the 
    actual observed value. Algorithm: After subtracting the two lists, the resulting list is multiplied by itself, and finally 
    all the values in the multiplied list are summed, and finally divide by the total number of list elements.

    Parameters
    ---------
        target_list: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

        predict_list: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    float. The mean square error  of two lists of one-dimensional data.

    Error
    ---------
        ValueError: The two lists must be the same length.
        Solution: Make sure that the number of numerical data in the two lists is the same (the same length).

        ValueError: List content must be one-dimensional numerical data: listA=",chack_A, "; listB=", chack_B.
        Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
        non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_A' is'True', it means that 'listA' does not need to be 
        changed; if it is'False', the input 'listA' is changed to a one-dimensional list of numerical data.ListB has the same judgment and processing 
        method as listA.

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' or 'listB' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    References
    ---------
    Mean square error: https://zh.wikipedia.org/wiki/%E5%86%B3%E5%AE%9A%E7%B3%BB%E6%95%B0

    """
    try:
        err = sub(target_list, predict_list)
        mse = sum(mult(err, err))
        return mse/len(target_list)
    except TypeError as err:
        raise TypeError(err)

def mean_absolute_error(target_list, predict_list) -> float:
    """Regression indicators for numerical prediction. Calculate the mean absolute error (MAE) of two lists of one-dimensional data.

     Mean absolute error  can measure the error between actual observations and predicted values. The calculation method is to 
     subtract the two lists, take the absolute value of each number and then add them all, and finally divide by the total number of list elements.

    Parameters
    ---------
        target_list: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

        predict_list: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    float. The root mean square error  of two lists of one-dimensional data.

    Error
    ---------
        ValueError: The two lists must be the same length.
        Solution: Make sure that the number of numerical data in the two lists is the same (the same length).

        ValueError: List content must be one-dimensional numerical data: listA=",chack_A, "; listB=", chack_B.
        Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
        non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_A' is'True', it means that 'listA' does not need to be 
        changed; if it is'False', the input 'listA' is changed to a one-dimensional list of numerical data.ListB has the same judgment and processing 
        method as listA.

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' or 'listB' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    References
    ---------
    MAE wiki :https://en.wikipedia.org/wiki/Mean_absolute_error

    """
    try:
        mae = sum(np.abs(sub(target_list, predict_list)))
        return mae/len(target_list)
    except TypeError as err:
        raise TypeError(err)

def root_mean_squard_error(target_list, predict_list) -> float:
    """Regression indicators for numerical prediction. Calculate the root mean square error (RMSE) of two lists of one-dimensional data.

    Root mean square error  can measure the error between actual observations and predicted values. 
    The calculation method is the square root of MSE.

    Parameters
    ---------
        target_list: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

        predict_list: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    float. The root mean square error  of two lists of one-dimensional data.

    Error
    ---------
        ValueError: The two lists must be the same length.
        Solution: Make sure that the number of numerical data in the two lists is the same (the same length).

        ValueError: List content must be one-dimensional numerical data: listA=",chack_A, "; listB=", chack_B.
        Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
        non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_A' is'True', it means that 'listA' does not need to be 
        changed; if it is'False', the input 'listA' is changed to a one-dimensional list of numerical data.ListB has the same judgment and processing 
        method as listA.

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' or 'listB' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    References
    ---------
    Root mean square  error(RMSE) wiki: https://en.wikipedia.org/wiki/Root-mean-square_deviation

    """
    try:
        mse = mean_square_error(target_list, predict_list)
        return mse**0.5
    except TypeError as err:
        raise TypeError(err)

def normalized_mean_squard_error(target_list, predict_list) -> float:
    """Regression indicators for numerical prediction. Calculate the normalized mean square error (NRMSE) of two lists of one-dimensional data.

    NRMSE is a statistical value, and its method is to normalize RMSE. The normalization method used by this function is that 
    the error between the maximum and minimum is used as the denominator, and RMSE is used as the numerator.

    The value is between 0 and 1. The closer the NRMSE is to 0, the smaller the error between the two and 
    the closer the model predicted value to the target value.

    Parameters
    ---------
        target_list: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

        predict_list: list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    float. The normalized mean square error  of two lists of one-dimensional data.

    Error
    ---------
        ValueError: The two lists must be the same length.
        Solution: Make sure that the number of numerical data in the two lists is the same (the same length).

        ValueError: List content must be one-dimensional numerical data: listA=",chack_A, "; listB=", chack_B.
        Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
        non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_A' is'True', it means that 'listA' does not need to be 
        changed; if it is'False', the input 'listA' is changed to a one-dimensional list of numerical data.ListB has the same judgment and processing 
        method as listA.

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' or 'listB' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    References
    ---------
    Normalized mean square error (NRMSE) wiki: https://zh.wikipedia.org/wiki/%E6%AD%A3%E8%A6%8F%E5%8C%96%E6%96%B9%E5%9D%87%E6%A0%B9%E5%B7%AE
    Normalized mean square error (NRMSE) function: https://www.marinedatascience.co/blog/2019/01/07/normalizing-the-rmse/

    """
    try:
        rmse = root_mean_squard_error(target_list, predict_list)
        return rmse/(max(predict_list) - min(predict_list))
    except TypeError as err:
        raise TypeError(err)

def ACF(listA, listB) -> float:
    """Calculate the autocorrelation coefficients of two lists of one-dimensional data.

    Autocorrelation coefficients: The autocorrelation as a function of the lag. 
    Algorithm: Calculate the autocorrelation coefficients according to the formula.

    Parameters
    ---------
        listA : list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

        listB : list ,ndarray, pandas.Series and pandas.DataFrame.
        One-dimensional numerical list.

    Returns
    ---------
    float. The autocorrelation coefficients of two lists of one-dimensional data.

    Error
    ---------
        ValueError: The two lists must be the same length.
        Solution: Make sure that the number of numerical data in the two lists is the same (the same length).

        ValueError: List content must be one-dimensional numerical data: listA=",chack_A, "; listB=", chack_B.
        Solution: 'True': one-dimensional numerical list. For example:[1,5,8,6,3].    'False': two-dimensional lists, strings, and one-dimensional 
        non-numerical lists.For example: [[3, 6], [7, 2, 9, 5]] , '5' ,  [3, 6, 7, '2', 9, 5] . If'chack_A' is'True', it means that 'listA' does not need to be 
        changed; if it is'False', the input 'listA' is changed to a one-dimensional list of numerical data.ListB has the same judgment and processing 
        method as listA.

        TypeError: object of type 'NoneType' has no len()
        Solution: 'listA' or 'listB' is None. Check that the input list is a one-dimensional list, for example: [1,5,8,6,3].

    References
    ---------
    Autocorrelation coefficients wiki: https://zh.wikipedia.org/wiki/%E8%87%AA%E7%9B%B8%E5%85%B3%E5%87%BD%E6%95%B0
    Autocorrelation Function: https://www.real-statistics.com/time-series-analysis/stochastic-processes/autocorrelation-function/

    """
    try:
        cov_A_B = covariance(listA, listB)
        sd_A = standard_deviation(listA)
        sd_B = standard_deviation(listB)
        P = cov_A_B / (sd_A * sd_B)
        return P
    except TypeError as err:
        raise TypeError(err)
