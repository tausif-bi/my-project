import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trendline_automation import fit_trendlines_single
import mplfinance as mpf

# Create an instance of the exchange (Binance in this example)
exchange = ccxt.binance({
    'enableRateLimit': True,
})

symbol = 'ETH/USDT'
timeframe = '5m'
limit = 100

# Fetch OHLCV data: [timestamp, open, high, low, close, volume]
ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

# Create DataFrame with appropriate column names
data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
data = data.set_index('date').astype(float)

# Drop any potential missing values (if any)
data = data.dropna()

# Set the lookback period
lookback = 72  # Ensure that your 100 candles is sufficient for your chosen lookback

# Compute trendlines and signals
def trendline_breakout(close: np.array, lookback:int):
    s_tl = np.full(len(close), np.nan)
    r_tl = np.full(len(close), np.nan)
    sig = np.zeros(len(close))

    for i in range(lookback, len(close)):
        # Select a window of the past 'lookback' candles
        window = close[i - lookback: i]

        # Fit trendlines using the helper function from trendline_automation.py
        s_coefs, r_coefs = fit_trendlines_single(window)

        # Project the trendlines to the current index
        s_val = s_coefs[1] + lookback * s_coefs[0]
        r_val = r_coefs[1] + lookback * r_coefs[0]

        s_tl[i] = s_val
        r_tl[i] = r_val

        # Generate signal: long if price breaks above resistance,
        # short if it breaks below support, else carry forward previous signal
        if close[i] > r_val:
            sig[i] = 1.0
        elif close[i] < s_val:
            sig[i] = -1.0
        else:
            sig[i] = sig[i - 1]

    return s_tl, r_tl, sig

# Run the breakout logic on the close prices
support, resist, signal = trendline_breakout(data['close'].to_numpy(), lookback)
data['support'] = support
data['resist'] = resist
data['signal'] = signal

# Plotting the results
plt.style.use('dark_background')
data['close'].plot(label='Close Price')
data['resist'].plot(label='Resistance', color='green')
data['support'].plot(label='Support', color='red')
plt.title("BTC/USDT 1h Trendline Breakout")
plt.legend()
plt.show()

# Compute returns for performance evaluation
# data['r'] = np.log(data['close']).diff().shift(-1)
# strat_r = data['signal'] * data['r']

# pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum() 
# print("Profit Factor", pf)

# strat_r.cumsum().plot()
# plt.ylabel("Cumulative Log Return")
# plt.title("Strategy Cumulative Returns")
# plt.show()
