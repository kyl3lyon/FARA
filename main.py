from flask import Flask, render_template, request, jsonify
from replit import web

# data manipulation and analysis libraries
import pandas as pd
from datetime import date

# data visualization libraries
import plotly.graph_objs as go
import plotly

# data retrieval and analysis libraries
import yfinance as yf

# time series analysis libraries
from pmdarima import auto_arima

# machine learning libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt


app = Flask(__name__)


def get_historical_data(tickers_list, start_date):
    today = date.today().strftime('%Y-%m-%d')
    result = pd.DataFrame()

    for ticker in tickers_list:
        yahoo_financials = yf.Ticker(ticker)
        data = yahoo_financials.history(start=start_date, end=today, interval='1d')
        data.index = data.index.normalize()

        data.rename(columns={'Open': f'{ticker}_open',
                             'Close': f'{ticker}_close',
                             'High': f'{ticker}_high',
                             'Low': f'{ticker}_low'}, inplace=True)

        result = pd.concat([result, data[[f'{ticker}_open', f'{ticker}_close', f'{ticker}_high', f'{ticker}_low']]], axis=1)

    return result


def calculate_indicators(df, ticker_list):

    def bollinger_bands(data, window=20, num_std=2):
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    for ticker in ticker_list:
        col = f'{ticker}_close'
        ticker_name = ticker.split('_')[0]  # extract ticker name

        if col not in df.columns:
            continue  # skip this iteration if the column does not exist in the dataframe

        # calculate 7-day and 21-day moving averages
        df[f'{ticker_name}_4w_mavg'] = df[col].rolling(window=7).mean()
        df[f'{ticker_name}_8w_mavg'] = df[col].rolling(window=21).mean()

        # calculate Bollinger Bands
        upper_band, lower_band = bollinger_bands(df[col])
        df[f'{ticker_name}_upper_band'] = upper_band
        df[f'{ticker_name}_lower_band'] = lower_band

        # calculate exponential moving average
        df[f'{ticker_name}_ema'] = df[col].ewm(span=12).mean()

        # calculate momentum
        df[f'{ticker_name}_momentum'] = df[col] - df[col].shift(1)

    # drop any rows with missing data
    df.dropna(inplace=True)

    return df


def arima_prediction_and_plot(assets, train_split=0.8, conf_int=0.95):
    fig = go.Figure()

    for asset_col, asset_indicators in assets.items():
        # get the asset_col column from the asset_indicators dataframe
        price_data = asset_indicators[asset_col]

        # calculate the train-test split index
        train_size = int(len(price_data) * train_split)

        # split the data into training and testing sets
        train_data = price_data[:train_size]
        test_data = price_data[train_size:]

        arima_model = auto_arima(train_data,
                                 seasonal=False,
                                 stepwise=True,
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

        fig.add_trace(go.Scatter(x=train_data.index, y=train_data,
                                 mode='lines',
                                 name=f'{asset_col} Training Data'))
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data,
                                 mode='lines',
                                 name=f'{asset_col} Test Data'))
        fig.add_trace(go.Scatter(x=test_data.index, y=forecast,
                                 mode='lines',
                                 name=f'{asset_col} Forecast'))
        fig.add_trace(go.Scatter(x=test_data.index, y=lower_series,
                                 mode='lines',
                                 name=f'{asset_col} Lower Bound',
                                 line=dict(width=0)))
        fig.add_trace(go.Scatter(x=test_data.index, y=upper_series,
                                 mode='lines',
                                 name=f'{asset_col} Upper Bound',
                                 fill='tonexty'))

    fig.update_layout(title='ARIMA Model Forecast for Asset(s) with 95% Confidence Interval',
                      xaxis_title='Date',
                      yaxis_title='Asset Price')

    return plotly.io.to_json(fig)

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

        plot = arima_prediction_and_plot(assets)
        return jsonify(plot=plot)  # Return the JSON representation of the plot
    return render_template('index.html')

if __name__ == '__main__':
    web.run(app)
