import argparse
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.dates import MinuteLocator, DateFormatter
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from time import sleep

def parse_args():
    parser = argparse.ArgumentParser(description='Stock Extrapolation Script')
    parser.add_argument('--nst', nargs='+', help='Nasdaq stock symbols', required=True)
    return parser.parse_args()

def calculate_percentage_trend(stock_data, baseline_trend, forecast_values):
    initial_price = stock_data.values[0]
    final_price = stock_data.values[-1]
    mean_price = stock_data.mean()
    trend_percentage = ((initial_price - final_price) / mean_price) * 100
    return trend_percentage

def download_stock_data(ticker, start_date, end_date, interval='1m'):
    # Download the stock data
    raw_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)['Close']

    # Filter out weekends and non-trading hours
    trading_days = raw_data.index.dayofweek < 5
    trading_hours = (raw_data.index.hour >= 9) & (raw_data.index.hour < 16)
    stock_data = raw_data[trading_days & trading_hours]

    return stock_data

def arima_extrapolation(stock_data, n_predict):
    n = len(stock_data)
    t = np.arange(0, n)

    # Fit a linear regression line to get the baseline trend
    model = LinearRegression().fit(t.reshape(-1, 1), stock_data)
    baseline_trend = model.predict(t.reshape(-1, 1))

    min_baseline_trend = baseline_trend - np.std(stock_data)
    max_baseline_trend = baseline_trend + np.std(stock_data)

    stock_detrended = stock_data - baseline_trend

    order = (5, 1, 0)  # Example order, you may need to fine-tune
    arima_model = ARIMA(stock_detrended, order=order)
    arima_result = arima_model.fit()

    forecast_values = arima_result.get_forecast(steps=n_predict).predicted_mean
    forecast_values += baseline_trend[-1]

    return forecast_values, baseline_trend, min_baseline_trend, max_baseline_trend

def plot_interpolation_and_extrapolation(ax, stock_data, forecast_values, baseline_trend, min_baseline_trend, max_baseline_trend, ticker):
    ax.clear()
    ax.set_facecolor('white')

    # Filter out weekends and non-trading hours for both plotting and calculations
    trading_days = stock_data.index.dayofweek < 5
    trading_hours = (stock_data.index.hour >= 9) & (stock_data.index.hour < 16)

    stock_data_filtered = stock_data[trading_days & trading_hours]
    baseline_trend_filtered = baseline_trend[trading_days & trading_hours]
    min_baseline_trend_filtered = min_baseline_trend[trading_days & trading_hours]
    max_baseline_trend_filtered = max_baseline_trend[trading_days & trading_hours]

    # Ensure forecast_values_filtered has the same length as stock_data_filtered
    forecast_values_filtered = forecast_values[-len(stock_data_filtered):]
    forecast_index = np.arange(len(stock_data_filtered), len(stock_data_filtered) + len(forecast_values_filtered))

    ax.plot(np.arange(len(stock_data_filtered)), stock_data_filtered.values, label=f'{ticker} Data', marker='.', color='black')
    ax.plot(forecast_index, forecast_values_filtered, linestyle='dashed', color='purple')
    ax.plot(np.arange(len(stock_data_filtered)), baseline_trend_filtered, linestyle='dotted', color='purple')
    ax.plot(np.arange(len(stock_data_filtered)), min_baseline_trend_filtered, linestyle='dotted', color='orange')
    ax.plot(np.arange(len(stock_data_filtered)), max_baseline_trend_filtered, linestyle='dotted', color='orange')

    ax.set_title(f'Stock Data and Extrapolation - {ticker}', color='black')
    ax.set_xlabel('Arbitrary Time Points', color='black')
    ax.set_ylabel('Stock Price', color='black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.axvline(len(stock_data_filtered)-1, color='gray', linestyle='--', linewidth=1, label='Extrapolation Start')

    trend_percentage = calculate_percentage_trend(stock_data_filtered, baseline_trend_filtered, forecast_values_filtered)
    ax.text(0.02, 0.05, f'Trend: {trend_percentage:.2f}%', transform=ax.transAxes, fontsize=8, color='black')



def update(frame, tickers, fig):
    if isinstance(fig, np.ndarray):
        # If fig is an array, it means multiple subplots
        axes = fig.flatten()
    else:
        # If fig is a single Axes object
        axes = [fig]

    end_date = pd.to_datetime('today')  # Current date
    start_date = end_date - pd.DateOffset(days=7)  # 7 days ago

    for i, (ticker, ax) in enumerate(zip(tickers, axes)):
        try:
            stock_data = download_stock_data(ticker, start_date, end_date, interval='1m')
            if not stock_data.empty:
                forecast_steps = 200
                forecast_values, baseline_trend, min_baseline_trend, max_baseline_trend = arima_extrapolation(stock_data.values, forecast_steps)
                shift = forecast_values[0] - stock_data[-1]
                plot_interpolation_and_extrapolation(ax, stock_data, forecast_values - shift, baseline_trend, min_baseline_trend, max_baseline_trend, ticker)
            else:
                # Clear the plot if data is not available
                ax.clear()
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")

def main():
    args = parse_args()
    tickers = args.nst
    num_tickers = len(tickers)

    if num_tickers > 1:
        fig, axes = plt.subplots(1, num_tickers, figsize=(15, 5))
        ani = FuncAnimation(fig, update, fargs=(tickers, axes), interval=2000)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ani = FuncAnimation(fig, update, fargs=(tickers, ax), interval=2000)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
