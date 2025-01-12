import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to fetch stock data
def rstock(ticker, period='5y'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    data.reset_index(inplace=True)
    return data[['Date', 'Close']]

def MC(data, num_simulations, num_days):
    # Calculate daily returns
    returns = data['Close'].pct_change().dropna()
    mean = returns.mean()
    std_dev = returns.std()

    # Initialize simulation
    simulations = np.zeros((num_days, num_simulations))
    initial_price = data['Close'].iloc[-1]
    simulations[0] = initial_price

    # Run simulations
    for t in range(1, num_days):
        random_shocks = np.random.normal(mean, std_dev, num_simulations)
        simulations[t] = simulations[t-1] * (1 + random_shocks)

    return simulations

def plot_simulation(data, simulations, title, ax):
    future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=simulations.shape[0], freq='B')
    
    for i in range(simulations.shape[1]):
        ax.plot(future_dates, simulations[:, i], lw=0.5, alpha=0.1, color='blue')
    
    # Calculate the percentiles for different confidence intervals
    p10 = np.percentile(simulations, 10, axis=1)
    p25 = np.percentile(simulations, 25, axis=1)
    p50 = np.percentile(simulations, 50, axis=1)
    p75 = np.percentile(simulations, 75, axis=1)
    p90 = np.percentile(simulations, 90, axis=1)
    
    # Plot the historical data
    ax.plot(data['Date'], data['Close'], label='Historical Data', color='black')
    
    # Plot the percentiles
    ax.plot(future_dates, p50, label='Median Forecast', color='red')
    ax.fill_between(future_dates, p10, p90, color='orange', alpha=0.3, label='10-90 percentile')
    ax.fill_between(future_dates, p25, p75, color='yellow', alpha=0.3, label='25-75 percentile')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

# Fetch stock data for a given ticker
ticker = 'VST'  # Example: Apple Inc.
data = rstock(ticker)

# Set parameters for simulation
num_simulations = 1000  # Number of simulation runs
forecast_periods = {
    '1-Day': 1,
    '5-Day': 5,
    '10-Day': 10,
    '3-Month': 90,
    '6-Month': 180,
    '2-Year': 730
}

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Run simulations and plot results in a grid layout
for i, (period_name, num_days) in enumerate(forecast_periods.items()):
    row = i // 2
    col = i % 2
    simulations = MC(data, num_simulations, num_days)
    plot_simulation(data, simulations, f'{period_name} Stock Price Forecast', axs[row, col])

plt.tight_layout()
plt.savefig("stock_forecast_grid.png")
plt.show()
