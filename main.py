from flask import Flask, render_template, request
from replit import web
import os

# data manipulation and analysis libraries
import pandas as pd
from datetime import date

# data visualization libraries
import matplotlib.pyplot as plt

# data retrieval and analysis libraries
import yfinance as yf

# time series analysis libraries
from pmdarima import auto_arima

# machine learning libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# img
import io
import base64

app = Flask(__name__)


def get_historical_data(tickers_list, start_date):
    today = date.today().strftime('%Y-%m-%d')
    result = pd.DataFrame()

    for ticker in tickers_list:
        yahoo_financials = yf.Ticker(ticker)
        data = yahoo_financials.history(start=start_date, end=today, interval='1wk')
        data.index = data.index.normalize()

        data.rename(columns={'Open': f'{ticker}_open',
                             'Close': f'{ticker}_close',
                             'High': f'{ticker}_high',
                             'Low': f'{ticker}_low'}, inplace=True)

        result = pd.concat([result, data[[f'{ticker}_open', f'{ticker}_close', f'{ticker}_high', f'{ticker}_low']]], axis=1)

    return result
  

# usage
tickers_list = ['AAPL', 'MSFT', 'INTC']
start_date = '2018-01-01'

historical_data = get_historical_data(tickers_list, start_date)


def calculate_indicators(df, ticker_list):
    
    def bollinger_bands(data, window=20, num_std=2):
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    for ticker in ticker_list:
        col = f'{ticker}_close'
        ticker_name = ticker.split('_')[0] # Extract ticker name

        # calculate 7-day and 21-day moving averages
        df[f'{ticker_name}_7d_mavg'] = df[col].rolling(window=4).mean()
        df[f'{ticker_name}_21d_mavg'] = df[col].rolling(window=8).mean()

        # calculate Bollinger Bands
        upper_band, lower_band = bollinger_bands(df[col])
        df[f'{ticker_name}_upper_band'] = upper_band
        df[f'{ticker_name}_lower_band'] = lower_band

        # calculate exponential moving average
        df[f'{ticker_name}_ema'] = df[col].ewm(span=12).mean()

        # calculate momentum
        df[f'{ticker_name}_momentum'] = df[col] - df[col].shift(1)

    # Drop any rows with missing data
    df.dropna(inplace=True)

    return df


def arima_prediction_and_plot(assets, train_split=0.8, conf_int=0.95):
    plt.figure(figsize=(12, 6))

    for asset_col, asset_indicators in assets.items():
        # Get the asset_col column from the asset_indicators dataframe
        price_data = asset_indicators[asset_col]

        # Calculate the train-test split index
        train_size = int(len(price_data) * train_split)

        # Split the data into training and testing sets
        train_data = price_data[:train_size]
        test_data = price_data[train_size:]

        arima_model = auto_arima(train_data, seasonal=False, stepwise=True,
                                 suppress_warnings=True,
                                 max_order=None, trace=True)

        arima_model.fit(train_data)

        forecast, conf_ints = arima_model.predict(n_periods=len(test_data), return_conf_int=True, alpha=(1 - conf_int))

        lower_series = pd.Series(conf_ints[:, 0], index=test_data.index)
        upper_series = pd.Series(conf_ints[:, 1], index=test_data.index)

        mae = mean_absolute_error(test_data, forecast)
        mse = mean_squared_error(test_data, forecast)
        rmse = sqrt(mse)

        print(f'{asset_col} MAE: {mae}')
        print(f'{asset_col} MSE: {mse}')
        print(f'{asset_col} RMSE: {rmse}')

        plt.plot(train_data, label=f'{asset_col} Training Data')
        plt.plot(test_data.index, test_data, label=f'{asset_col} Test Data')
        plt.plot(test_data.index, forecast, label=f'{asset_col} Forecast')
        plt.fill_between(test_data.index, lower_series, upper_series, alpha=0.2)

    plt.xlabel('Date')
    plt.ylabel('Asset Price')
    plt.title('ARIMA Model Forecast for Asset(s) with 95% Confidence Interval')
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        tickers_list = request.form.getlist('tickers')
        start_date = request.form['start_date']

        historical_data = get_historical_data(tickers_list, start_date)
        asset_indicators = calculate_indicators(historical_data, tickers_list)

        assets_to_analyze = [col for col in historical_data.columns if col.endswith('_close')]
        assets = {}

        for asset_col in assets_to_analyze:
            asset_indicators = calculate_indicators(historical_data, asset_col)
            assets[asset_col] = asset_indicators

        img = arima_prediction_and_plot(assets)
        return render_template('index.html', plot=img)
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

web.run(app)