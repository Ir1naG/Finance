import argparse
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.dates import MinuteLocator, DateFormatter
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

def fourier_extrapolation(stock_data, n_predict):
    n = len(stock_data)
    t = np.arange(0, n)
    
    # Fit a linear regression line to get the baseline trend
    model = LinearRegression().fit(t.reshape(-1, 1), stock_data)
    baseline_trend = model.predict(t.reshape(-1, 1))

    # Calculate two additional baseline trends representing possible errors (min and max)
    min_baseline_trend = baseline_trend - np.std(stock_data)
    max_baseline_trend = baseline_trend + np.std(stock_data)

    stock_detrended = stock_data - baseline_trend

    f = np.fft.fft(stock_detrended)
    f[n_predict:] = 0
    stock_extrapolated = np.fft.ifft(f).real

    forecast_values = stock_extrapolated[-n_predict:] + baseline_trend[-1]

    return forecast_values, baseline_trend, min_baseline_trend, max_baseline_trend

def plot_interpolation_and_extrapolation(ax, stock_data, forecast_values, baseline_trend, min_baseline_trend, max_baseline_trend, ticker):
    ax.clear()
    ax.set_facecolor('white')  # Set black background
    ax.plot(stock_data.index, stock_data.values, label=f'{ticker} Data', marker='.', color='black')
    forecast_index = pd.date_range(start=stock_data.index[-1], periods=len(forecast_values) + 1, freq='T')[1:]
    ax.plot(forecast_index, forecast_values, linestyle='dashed', color='purple')
    ax.plot(stock_data.index, baseline_trend, linestyle='dotted', color='purple')
    ax.plot(stock_data.index, min_baseline_trend, linestyle='dotted', color='orange')
    ax.plot(stock_data.index, max_baseline_trend, linestyle='dotted', color='orange')

    ax.set_title(f'Stock Data and Extrapolation - {ticker}', color='black')
    ax.set_xlabel('Time', color='black')
    ax.set_ylabel('Stock Price', color='black')
    ax.xaxis.set_major_locator(MinuteLocator(interval=30))  # Set major locator every 30 minutes
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))  # Format the x-axis as hours:minutes
    ax.tick_params(axis='x', rotation=90, colors='black')  # Rotate x-axis labels by 90 degrees
    ax.tick_params(axis='y', colors='black')
    ax.axvline(stock_data.index[-1], color='gray', linestyle='--', linewidth=1, label='Extrapolation Start')

    # Calculate the percentage trend based on the first and last points of the linear regression
    trend_percentage = calculate_percentage_trend(stock_data, baseline_trend, forecast_values)

    # Add the percentage trend as a label on the bottom left
    ax.text(0.02, 0.05, f'Trend: {trend_percentage:.2f}%', transform=ax.transAxes, fontsize=8, color='black')

def update(frame, tickers, axes):
    for i, (ticker, ax) in enumerate(zip(tickers, axes)):
        stock_data = yf.download(ticker, period='1d', interval='1m')['Close']#[-360:]  # Last 6 hours
        forecast_steps = 200
        forecast_values, baseline_trend, min_baseline_trend, max_baseline_trend = fourier_extrapolation(stock_data.values, forecast_steps)
        shift = forecast_values[0] - stock_data[-1]
        plot_interpolation_and_extrapolation(ax, stock_data, forecast_values - shift, baseline_trend, min_baseline_trend, max_baseline_trend, ticker)

def main():
    args = parse_args()
    tickers = args.nst
    num_tickers = len(tickers)

    fig, axes = plt.subplots(1, num_tickers, figsize=(15, 5))
    ani = FuncAnimation(fig, update, fargs=(tickers, axes), interval=2000)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
