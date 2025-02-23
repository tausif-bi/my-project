import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trendline_automation import fit_trendlines_single
import mplfinance as mpf

# --------------------------
# Data fetching using ccxt
# --------------------------
exchange = ccxt.binance({'enableRateLimit': True})
symbol = 'ETH/USDT'
timeframe = '5m'
limit = 100

# Fetch OHLCV data: [timestamp, open, high, low, close, volume]
ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

# Create a DataFrame with proper column names
data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
data = data.set_index('date').astype(float)
data = data.dropna()

# --------------------------
# Trendline breakout function
# --------------------------
def trendline_breakout(close: np.array, lookback: int):
    # Initialize arrays for support trendline, resistance trendline
    s_tl = np.full(len(close), np.nan)
    r_tl = np.full(len(close), np.nan)

    # Initialize signals
    buy_sig = np.zeros(len(close))          # Crossing above support
    res_break_sig = np.zeros(len(close))    # Crossing above resistance

    for i in range(lookback, len(close)):
        # Use the past 'lookback' candles to compute trendlines
        window = close[i - lookback: i]

        # Calculate support and resistance trendline coefficients
        s_coefs, r_coefs = fit_trendlines_single(window)

        # Project the trendlines to the current index
        s_val = s_coefs[1] + lookback * s_coefs[0]
        r_val = r_coefs[1] + lookback * r_coefs[0]

        s_tl[i] = s_val
        r_tl[i] = r_val

        # 1) Crossing Above Support
        if i > lookback:
            # Check if the previous close was below/at support and current is above
            if (close[i - 1] <= s_tl[i - 1]) and (close[i] > s_val):
                buy_sig[i] = 1.0  # Buy signal triggered

        # 2) Crossing Above Resistance
        if i > lookback:
            # Check if the previous close was below/at resistance and current is above
            if (close[i - 1] <= r_tl[i - 1]) and (close[i] > r_val):
                res_break_sig[i] = 1.0

    return s_tl, r_tl, buy_sig, res_break_sig

# --------------------------
# Execute breakout logic
# --------------------------
lookback = 72
support, resist, buy_signal, res_break_signal = trendline_breakout(data['close'].to_numpy(), lookback)
data['support'] = support
data['resist'] = resist
data['buy_signal'] = buy_signal
data['res_break_signal'] = res_break_signal

# --------------------------
# Plotting the results
# --------------------------
plt.style.use('dark_background')
plt.figure(figsize=(12, 6))
data['close'].plot(label='Close Price')
data['support'].plot(label='Support Trendline', color='red')
data['resist'].plot(label='Resistance Trendline', color='green')

# Mark the buy signals on the chart (support breakout)
buy_indices = data.index[data['buy_signal'] == 1.0]
plt.scatter(buy_indices.to_pydatetime(), data.loc[buy_indices, 'close'],
            marker='^', color='lime', s=100, label='Support Breakout (Buy)')

# Mark the resistance break signals on the chart
res_break_indices = data.index[data['res_break_signal'] == 1.0]
plt.scatter(res_break_indices.to_pydatetime(), data.loc[res_break_indices, 'close'],
            marker='^', color='yellow', s=100, label='Resistance Breakout')

plt.title("1h Trendline Breakout with Buy Signals and Resistance Break Signals")
plt.legend()
plt.show()
