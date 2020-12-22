# TimesML
---
## About
This package was developed for time series data analysis and machine learning tasks. The aim of TimesML is to provide high-level APIs that developers and data scientists can easily model their times series data. We plan to support more machine learning models in the future. Thank you for your support, please star⭐ this project if you like.

## List of module
#### Math
* Statistics

    This module contains statistics calculation function.

    ex. mean_square_error、coefficient_of_determination、
        ACF

#### TimeSeriesAnalysis
* Chart

    Time series data related drawing function.

    ex. statistics_infographic、ACF_chart、forecast_result_group_chart

* Model

    This module contains time series models for forecasting.

    ex. AutoRegressive、SimpleMovingAverage

* ProcessData

    This module contains the access and processing methods of time series data.

    ex. get_data_yahoo、n_order_difference_data、split_data


## Simple example
```js
import TimeSeriesAnalysis.ProcessData as Data
import TimeSeriesAnalysis.Model as Model
import TimeSeriesAnalysis.Chart as Chart

# setting parameters
save_path = 'US'
chart = Chart.chart('US')

# read data
data = Data.read_file(path='test_data/g20_new_c.csv', col_name='US')

# contains basic statistics: historocal trend line chart、lag plot、ACF chart. figure 1.
chart.statistics_infographic(data, file_path=save_path, lags=20, xlabel='date', ylabel='population')

# split data into training set and test set
train, test = Data.split_data(data, ratio=0.7)

# autoRegressive
model = Model.AutoRegressive(lags=2)
model.fit(train)
model.predict(test, pure_test_set_predict=True)

model1 = Model.AutoRegressive(lags=20)
model1.fit(train)
model1.predict(test,pure_test_set_predict= True)

# Save data
Data.save_flie(model1.test_predict, path=save_path, stock_id='US', file_format='csv')

# Combine and compare the prediction results of the two models. figure 2.
chart.forecast_result_group_chart(train, test, model, model1, file_path=save_path, 
model_1_name='AR(2)', model_2_name='AR(20)', xlabel='date', ylabel='population')

# simple moving average
model2 = Model.SimpleMovingAverage(windows=5)
model2.fit(data)

# line chart to observe the average situation every five days. figure 3.
chart.line_chart(data, model2.sma_result, chart_title='SMA(5)', file_path=save_path, xlabel='date', ylabel='price')
```
## Chart example
#### Figure 1: chart.statistics_infographic
![image](https://github.com/leodflag/TimesML/blob/master/US/statistics_infographic_US.png)
    historocal trend line chart(Draw historical trend line chart with time series data.)、
    lag plot(Scatter plot of lagging periods.)、
    ACF chart( Autocorrelation coefficient chart.).

#### Figure 2: chart.forecast_result_group_chart
![image](https://github.com/leodflag/TimesML/blob/master/US/forecast_result_group_chart_US.png)
    Combine and compare the prediction results of the two models.

#### Figure 3: chart.line_chart
![image](https://github.com/leodflag/TimesML/blob/master/US/SMA(5)_US_line_chart.png)
[TimesML (github)](https://github.com/leodflag/TimesML)
    line chart to observe the average situation every five days.
