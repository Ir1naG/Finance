!pip install mplfinance
!pip install statsmodels
!pip install pmdarima 
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf

# For Auto-ARIMA
from pmdarima import auto_arima
import warnings

warnings.filterwarnings("ignore")  # Suppress most convergence warnings


def download_stock_data(ticker, period="1y", interval="1d", debug=True):
    """
    Downloads stock data via yfinance and performs basic validations.
    
    :param ticker:   Ticker symbol, e.g., 'AAPL'
    :param period:   How far back to fetch data ('1y', '6mo', '2y', etc.)
    :param interval: Granularity ('1d', '1wk', '1mo')
    :param debug:    Whether to print debug info
    :return:         Cleaned DataFrame with market data
    """
    if debug:
        print(f"\n[DEBUG] Downloading {ticker} data: period={period}, interval={interval}")

    df = yf.download(ticker, period=period, interval=interval)

    # Quick check of returned data
    if debug:
        print(f"[DEBUG] Downloaded rows: {len(df)}")
        print("[DEBUG] DataFrame info:")
        print(df.info())
        print("\n[DEBUG] Checking head of DataFrame:")
        print(df.head())

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if debug:
            print("[DEBUG] Converting index to datetime...")
        df.index = pd.to_datetime(df.index)

    # Drop rows where 'Close' is missing
    df = df.dropna(subset=['Close'])

    if debug:
        print(f"\n[DEBUG] Data after cleaning, final rows: {len(df)}")
        print(df[['Close']].tail())

    return df


def find_support_resistance(df, window=5):
    """
    Computes approximate support/resistance by looking for local minima/maxima
    in a rolling window.
    
    :param df:     DataFrame containing 'High' and 'Low'
    :param window: Rolling window size
    :return:       (supports, resistances) as lists of (Date, Price)
    """
    rolling_min = df['Low'].rolling(window=window, center=True).min()
    rolling_max = df['High'].rolling(window=window, center=True).max()

    supports = []
    resistances = []

    # Identify potential support: 'Low' equals rolling_min
    # Identify potential resistance: 'High' equals rolling_max
    for i in range(window // 2, len(df) - window // 2):
        if df['Low'].iloc[i] == rolling_min.iloc[i] and not np.isnan(rolling_min.iloc[i]):
            supports.append((df.index[i], df['Low'].iloc[i]))
        if df['High'].iloc[i] == rolling_max.iloc[i] and not np.isnan(rolling_max.iloc[i]):
            resistances.append((df.index[i], df['High'].iloc[i]))

    return supports, resistances


def moving_average_crossover_signal(df, short_window=20, long_window=50):
    """
    Simple bullish/bearish signal based on MA crossover.
    
    :param df:            DataFrame with 'Close'
    :param short_window:  Window for short SMA
    :param long_window:   Window for long SMA
    :return:              "Bullish", "Bearish" or "Neutral"/"Not enough data"
    """
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long']  = df['Close'].rolling(window=long_window).mean()

    short_latest = df['SMA_short'].iloc[-1]
    long_latest = df['SMA_long'].iloc[-1]

    if np.isnan(short_latest) or np.isnan(long_latest):
        return "Not enough data"

    if short_latest > long_latest:
        return "Bullish"
    elif short_latest < long_latest:
        return "Bearish"
    else:
        return "Neutral"


def print_support_resistance(supports, resistances):
    """
    Utility to print the last few support/resistance levels found.
    """
    if supports:
        print("Support Levels:")
        for date, level in supports[-5:]:
            print(f"  {date.strftime('%Y-%m-%d')} -> {level:.2f}")
    else:
        print("No obvious support levels found.")

    if resistances:
        print("\nResistance Levels:")
        for date, level in resistances[-5:]:
            print(f"  {date.strftime('%Y-%m-%d')} -> {level:.2f}")
    else:
        print("No obvious resistance levels found.")


def forecast_with_autoarima(ts, steps=30, debug=True):
    """
    Uses Auto-ARIMA to find the best (p, d, q) for the 'Close' price
    and forecasts the next `steps` business days.
    
    :param ts:    Pandas Series (time-indexed) of the 'Close' prices
    :param steps: Number of days to forecast
    :param debug: Whether to print debug logs
    :return:      A pandas Series with forecasted values (or None if fitting fails)
    """
    if len(ts) < 30:
        if debug:
            print("[DEBUG] Not enough data to build a meaningful model (need >30 data points).")
        return None

    # Attempt Auto-ARIMA fitting
    if debug:
        print("[DEBUG] Initiating auto_arima with trace=True...")
    try:
        model = auto_arima(
            ts,
            start_p=1, start_q=1,
            max_p=5, max_q=5,
            d=None,            # Let model figure out differencing
            seasonal=False,    # No daily-seasonal logic here
            trace=True,        # Print logs of the fitting process
            error_action='ignore',
            suppress_warnings=False,
            stepwise=True
        )
    except Exception as e:
        if debug:
            print(f"[DEBUG] auto_arima fitting failed: {e}")
        return None

    if debug:
        print("\n[DEBUG] ARIMA model summary:")
        print(model.summary())

    # Forecast next 'steps' points
    try:
        forecast_values = model.predict(n_periods=steps)
    except Exception as e:
        if debug:
            print(f"[DEBUG] Forecasting failed: {e}")
        return None

    # Create date range for the forecast
    last_date = ts.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                   periods=steps,
                                   freq='B')  # 'B' = business days

    forecast_series = pd.Series(forecast_values, index=forecast_index, name='Forecast')
    return forecast_series


def analyze_stock_debug(ticker, debug=True):
    """
    Comprehensive analysis:
      - Download daily/weekly/monthly data
      - Print support/resistance
      - Print bullish/bearish signals
      - Auto-ARIMA forecast on daily data
      - Plots candlestick charts
    """

    # 1) Retrieve data for multiple timeframes
    if debug:
        print(f"\n[DEBUG] Analyzing {ticker} in daily, weekly, monthly timeframes...")

    daily_df = download_stock_data(ticker, period="6mo", interval="1d", debug=debug)
    weekly_df = download_stock_data(ticker, period="2y", interval="1wk", debug=debug)
    monthly_df = download_stock_data(ticker, period="5y", interval="1mo", debug=debug)

    # If no data in daily_df, we can't proceed with forecasting or signals
    if daily_df.empty:
        print("[DEBUG] Daily data is empty. Aborting analysis.")
        return

    # 2) Calculate Support/Resistance
    if not daily_df.empty:
        daily_supports, daily_resistances = find_support_resistance(daily_df)
    if not weekly_df.empty:
        weekly_supports, weekly_resistances = find_support_resistance(weekly_df)
    if not monthly_df.empty:
        monthly_supports, monthly_resistances = find_support_resistance(monthly_df)

    # 3) Print out Support/Resistance
    print("\n=== Daily Data (6mo) Support/Resistance ===")
    print_support_resistance(daily_supports, daily_resistances)

    print("\n=== Weekly Data (2y) Support/Resistance ===")
    print_support_resistance(weekly_supports, weekly_resistances)

    print("\n=== Monthly Data (5y) Support/Resistance ===")
    print_support_resistance(monthly_supports, monthly_resistances)

    # 4) Bullish/Bearish Signals
    daily_signal = moving_average_crossover_signal(daily_df)
    weekly_signal = moving_average_crossover_signal(weekly_df)
    monthly_signal = moving_average_crossover_signal(monthly_df)

    print("\n=== Bullish/Bearish Signals ===")
    print(f"Daily signal:   {daily_signal}")
    print(f"Weekly signal:  {weekly_signal}")
    print(f"Monthly signal: {monthly_signal}")

    # 5) Forecast on daily data
    print("\n=== Forecasting the Next 30 Days (Daily) ===")
    daily_ts = daily_df['Close']  # isolate the 'Close' Series
    forecast_series = forecast_with_autoarima(daily_ts, steps=30, debug=debug)
    if forecast_series is not None:
        print(f"\n[DEBUG] Forecast results:")
        print(forecast_series)
    else:
        print("[DEBUG] Forecast returned None. No predictions available.")

    # 6) Plot candlestick charts (daily, weekly, monthly)
    if not daily_df.empty:
        mpf.plot(daily_df, type='candle', volume=True,
                 title=f"{ticker} - Daily (6mo)")
    if not weekly_df.empty:
        mpf.plot(weekly_df, type='candle', volume=True,
                 title=f"{ticker} - Weekly (2y)")
    if not monthly_df.empty:
        mpf.plot(monthly_df, type='candle', volume=True,
                 title=f"{ticker} - Monthly (5y)")


if __name__ == "__main__":
    # Run analysis in debug mode (True). Set to False to reduce logging output.
    analyze_stock_debug("AAPL", debug=True)
