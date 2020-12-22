#!/usr/bin/env python
# -*- coding: utf-8 -*-
import TimeSeriesAnalysis.ProcessData as Data
import TimeSeriesAnalysis.Model as Model
import TimeSeriesAnalysis.Chart as Chart

save_path = '2033.TW'
chart = Chart.chart('2033.TW')

Data.get_data_yahoo(stock_id='2033.TW', start_period='2019, 1, 1', end_period=' 2020, 1, 1', file_format='csv', frequency='day')
data = Data.read_file(path='2033.TW/2033.TW_day.csv', col_name='close')

chart.statistics_infographic(data, file_path=save_path, lags=20, xlabel='date', ylabel='price')

train, test = Data.split_data(data, ratio=0.7)

model = Model.AutoRegressive(lags=2)
model.fit(train)
model.predict(test, pure_test_set_predict=True)

model1 = Model.AutoRegressive(lags=20)
model1.fit(train)
model1.predict(test,pure_test_set_predict= True)
Data.save_flie(model1.test_predict, path=save_path, stock_id='2033.TW', file_format='csv')

chart.forecast_result_group_chart(train, test, model, model1, file_path=save_path, 
model_1_name='AR(2)', model_2_name='AR(20)', xlabel='date', ylabel='price')

model2 = Model.SimpleMovingAverage(windows=5)
model2.fit(data)
chart.line_chart(data, model2.sma_result, chart_title='SMA(5)', file_path=save_path, xlabel='date', ylabel='price')
