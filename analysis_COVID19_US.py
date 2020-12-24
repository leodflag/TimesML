import TimeSeriesAnalysis.ProcessData as Data
import TimeSeriesAnalysis.Model as Model
import TimeSeriesAnalysis.Chart as Chart

# setting parameters
save_path = 'US'
chart = Chart.chart('US')

# read data
data = Data.read_file(path='test_data/g20_new_c.csv', col_name='US')
chart.lag_plot(data, save_path)

# contains basic statistics: historocal trend line chart、lag plot、ACF chart. figure 1.
chart.statistics_infographic(data, file_path=save_path, lags=20, xlabel='date', ylabel='population')

# split data into training set and test set
train, test = Data.split_data(data, ratio=0.7)

# autoregressive lag periods :2
model1 = Model.AutoRegressive(lags=2)
model1.fit(train)
model1.predict(test, pure_test_set_predict=True)

# autoregressive lag periods :20
model2 = Model.AutoRegressive(lags=20)
model2.fit(train)
model2.predict(test,pure_test_set_predict= True)

# Save the data predicted by model1 using the test set
Data.save_flie(model1.test_predict, path=save_path, stock_id='US', file_format='csv')

# Combine and compare the prediction results of the two models. figure 2.
chart.forecast_result_group_chart(train, test, model1, model2, file_path=save_path, 
model_1_name='AR(2)', model_2_name='AR(20)', xlabel='date', ylabel='population')

# simple moving average. five days as a unit to calculate the average
model3 = Model.SimpleMovingAverage(windows=5)
model3.fit(data)

# line chart to observe the average situation every five days. figure 3.
chart.line_chart(data, model3.sma_result, chart_title='SMA(5)', file_path=save_path, xlabel='date', ylabel='price')
