# Financial ARIMA Forecast for Multiple Assets
This repository contains a Flask web application designed for stock analysis and forecasting using the ARIMA (AutoRegressive Integrated Moving Average) model. The application allows users to input multiple stock tickers and a start date, after which it retrieves historical stock data, calculates various technical indicators, and generates stock price forecasts using the ARIMA model.

## How it works
The application is built using the Flask web framework and leverages the Yahoo Finance API for retrieving historical stock data. The user interface consists of a simple form that accepts a comma-separated list of stock tickers and a start date. Upon submission, the application processes the user's input and performs the following steps:

1. **Data Retrieval**: The application fetches historical data for the specified stock tickers using the Yahoo Finance API. The historical data includes the Open, Close, High, and Low prices for each trading day within the specified date range.

2. **Indicator Calculation**: The application calculates various technical indicators for each stock ticker, such as:
    * 7-day and 21-day Moving Averages
    * Bollinger Bands
    * Exponential Moving Averages
    * Momentum
    
    These indicators help in understanding the stock's historical price movements and can be useful in identifying trends and potential trading opportunities.

3. **ARIMA Forecasting**: The application employs the ARIMA model to forecast stock prices. It uses the PMDARIMA library to automatically select the optimal model parameters based on the training data. The training data consists of a portion of the historical data (default: 80%), while the remaining data is used for testing the model's performance. The model then generates forecasts for the test data along with a 95% confidence interval.

4. **Data Visualization**: The application uses the Plotly library to visualize the historical data, technical indicators, and ARIMA forecasts. The interactive charts allow users to analyze the stock's historical performance and the accuracy of the ARIMA model's predictions.

Through this process, the application provides an efficient and user-friendly way to analyze stocks and generate price forecasts using the ARIMA model, **however, no one should use it for investment decisions** as this was just a learning exercise for myself.
