# TimesML
---
## About
This package is used for time series data analysis. In the future, analysis methods such as statistical models and machine learning models will be incorporated so that users can easily use them.

## List of module
#### Math
* Statistics

    This module contains statistics calculation function.

#### TimeSeriesAnalysis
* Chart

    Time series data related drawing function.

* Model

    This module contains time series models for forecasting.

* ProcessData

    This module contains the access and processing methods of time series data.


## Simple example
```js
    import TimeSeriesAnalysis.ProcessData as Data
    import TimeSeriesAnalysis.Chart as Chart
    import TimeSeriesAnalysis.Model as model
    import Math.Statistics as math

    lags = 2
    col_name = 'close'
    file_path = 'test_data/TWD_unittest.csv'
    stock_id = 'TWD'

    # TimeSeriesAnalysis.ProcessData
    data = Data.read_file(file_path, col_name)
    train, test = Data.split_data(data, 0.7)

    #TimeSeriesAnalysis.Chart
    chart = Chart.chart(stock_id)
    chart.historocal_trend_line_chart(data)

    # TimeSeriesAnalysis.Model
    model = model.AutoRegressive(lags)
    model.fit(train)
    model.predict(test)
    print(model.mse)  # 0.12240460258063346
```
## Chart example
![image](https://github.com/leodflag/TimesML/blob/master/TWD/Historocal_Trend_TWD.png)

[TimesML (github)](https://github.com/leodflag/TimesML)
