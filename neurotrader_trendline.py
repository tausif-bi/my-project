import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

# -------------------------------
# 1. Fetch Data via ccxt
# -------------------------------
exchange = ccxt.binance({'enableRateLimit': True})
symbol = 'ETH/USDT'
timeframe = '1h'
limit = 500  # Number of daily candles to fetch
ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

# Convert OHLCV data into a DataFrame
data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
data = data.set_index('date')
data = data.sort_index()

# Take the natural logarithm of the price columns to handle scaling issues
data[['open', 'high', 'low', 'close']] = np.log(data[['open', 'high', 'low', 'close']])

# -------------------------------
# 2. Trendline Functions
# -------------------------------
def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    # Compute the intercept so that the line goes through the pivot point.
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
    diffs = line_vals - y

    # For a support line, the line should lie below the prices;
    # for a resistance line, above.
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Return the squared sum of differences (error)
    err = (diffs ** 2.0).sum()
    return err

def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.array):
    # Determine a base unit to adjust the slope.
    slope_unit = (y.max() - y.min()) / len(y)
    
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step

    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert(best_err >= 0.0)  # The initial slope must be valid

    get_derivative = True
    derivative = None
    while curr_step > min_step:
        if get_derivative:
            # Numerical differentiation to decide the direction of slope change
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err
            
            # Try the opposite direction if needed
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0:
                raise Exception("Derivative failed. Check your data.")

            get_derivative = False

        # Adjust the slope based on the derivative
        if derivative > 0.0:
            test_slope = best_slope - slope_unit * curr_step
        else:
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5  # Reduce the step size if no improvement
        else:
            best_err = test_err
            best_slope = test_slope
            get_derivative = True  # Recompute derivative for next iteration

    return best_slope, -best_slope * pivot + y[pivot]

def fit_trendlines_single(data_series: np.array):
    # Compute a line of best fit using least squares.
    x = np.arange(len(data_series))
    coefs = np.polyfit(x, data_series, 1)
    line_points = coefs[0] * x + coefs[1]

    # Identify the pivots based on deviations from the line.
    upper_pivot = (data_series - line_points).argmax()
    lower_pivot = (data_series - line_points).argmin()

    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data_series)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data_series)
    return support_coefs, resist_coefs

def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    # Use the close prices for the best fit line, then adjust using high and low.
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    line_points = coefs[0] * x + coefs[1]

    upper_pivot = (high - line_points).argmax()
    lower_pivot = (low - line_points).argmin()

    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)
    return support_coefs, resist_coefs

# -------------------------------
# 3. (Optional) Compute Trendline Slopes on Entire Data
# -------------------------------
lookback = 60
support_slope = [np.nan] * len(data)
resist_slope = [np.nan] * len(data)

for i in range(lookback - 1, len(data)):
    candles_window = data.iloc[i - lookback + 1: i + 1]
    support_coefs, resist_coefs = fit_trendlines_high_low(
        candles_window['high'], candles_window['low'], candles_window['close']
    )
    support_slope[i] = support_coefs[0]
    resist_slope[i] = resist_coefs[0]

data['support_slope'] = support_slope
data['resist_slope'] = resist_slope

# -------------------------------
# 4. Plot Trendlines on Candles with mplfinance
# -------------------------------
# Use the last 30 candles for visualization.
candles = data.iloc[-60:].copy()

# Compute trendlines using both the single and high/low methods.
support_coefs_c, resist_coefs_c = fit_trendlines_single(candles['close'])
support_coefs, resist_coefs = fit_trendlines_high_low(candles['high'], candles['low'], candles['close'])

# Generate the line values
support_line_c = support_coefs_c[0] * np.arange(len(candles)) + support_coefs_c[1]
resist_line_c = resist_coefs_c[0] * np.arange(len(candles)) + resist_coefs_c[1]
support_line = support_coefs[0] * np.arange(len(candles)) + support_coefs[1]
resist_line = resist_coefs[0] * np.arange(len(candles)) + resist_coefs[1]

# Prepare a function to convert line values into tuples for mplfinance.
def get_line_points(candles_df, line_points):
    idx = candles_df.index
    line_i = len(candles_df) - len(line_points)
    assert(line_i >= 0)
    points = []
    for i in range(line_i, len(candles_df)):
        points.append((idx[i], line_points[i - line_i]))
    return points

s_seq = get_line_points(candles, support_line)
r_seq = get_line_points(candles, resist_line)
s_seq2 = get_line_points(candles, support_line_c)
r_seq2 = get_line_points(candles, resist_line_c)

# Plot candles and overlay the trendlines
plt.style.use('dark_background')
ax = plt.gca()  # Get current axes for mplfinance
mpf.plot(
    candles,
    type='candle',
    style='charles',
    alines=dict(
        alines=[s_seq, r_seq, s_seq2, r_seq2],
        colors=['w', 'w', 'b', 'b']  # You can adjust colors as desired.
    ),
    ax=ax
)
plt.show()
