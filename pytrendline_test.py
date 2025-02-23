import ccxt
import pandas as pd
import pytrendline
from pytrendline import CandlestickData


# Initialize the exchange (example with Binance)
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1h'
limit = 100  # Fetch the last 100 candles


# Fetch OHLCV data from ccxt
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
# Convert to DataFrame; ccxt returns [timestamp, open, high, low, close, volume]
df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
# Convert timestamp to datetime
df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')

# Create a CandlestickData object for pytrendline
candlestick_data = pytrendline.CandlestickData(
    df=df,
    time_interval=timeframe,    # e.g. "1m"
    open_col="Open",
    high_col="High",
    low_col="Low",
    close_col="Close",
    datetime_col="Date"         # use this if the datetime is in a column (not index)
)

# Detect trendlines (support/resistance)
results = pytrendline.detect(
    candlestick_data=candlestick_data,
    trend_type=pytrendline.TrendlineTypes.BOTH,  # Both support and resistance
    first_pt_must_be_pivot=False,
    last_pt_must_be_pivot=False,
    all_pts_must_be_pivots=False,
    trendline_must_include_global_maxmin_pt=False,
    min_points_required=3,
    scan_from_date=None,
    ignore_breakouts=True,
    config={}
)

# Optionally, plot the trendlines
pytrendline.plot(
    results=results,
    filedir='.',
    filename='output.html'
)
