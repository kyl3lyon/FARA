from flask import Flask, render_template, request, jsonify
from replit import web

app = Flask(__name__)

from preprocessing import get_historical_data, calculate_indicators, arima_prediction_and_plot

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
