import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trendline_automation import fit_trendlines_single
import mplfinance as mpf
import schedule
import time
import requests
import matplotlib.dates as mdates

# ----------------------------------
# Discord Webhook Configuration
# ----------------------------------
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1342868833245073448/JcPbZr57wAOneJrKUO0PueBRI0my0SGu7YoIpyhs-5rtRsgTAO6_IcHLR45VqNu9TW8O"

def send_discord_message(message: str):
    """Send a text message to Discord via webhook."""
    payload = {
        "content": message
    }
    requests.post(DISCORD_WEBHOOK_URL, json=payload)

# ----------------------------------
# Trendline breakout function
# ----------------------------------
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

def fetch_and_check_breakouts():
    """
    Fetch latest data, compute trendlines, detect breakouts on the most recent candle,
    and send a Discord notification if breakouts occur.
    """
    # --------------------------
    # Data fetching using ccxt
    # --------------------------
    exchange = ccxt.binance({'enableRateLimit': True})
    symbol = 'ETH/USDT'
    timeframe = '1h'
    limit = 100

    # Fetch OHLCV data: [timestamp, open, high, low, close, volume]
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    # Create a DataFrame with proper column names
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data.set_index('date').astype(float)
    data = data.dropna()

    # Run the breakout logic
    lookback = 72
    support, resist, buy_signal, res_break_signal = trendline_breakout(data['close'].to_numpy(), lookback)
    data['support'] = support
    data['resist'] = resist
    data['buy_signal'] = buy_signal
    data['res_break_signal'] = res_break_signal

    # Check the most recent candle
    latest_index = data.index[-1]
    latest_buy_sig = data['buy_signal'].iloc[-1]
    latest_res_sig = data['res_break_signal'].iloc[-1]

    # If there's a buy signal or a resistance breakout signal, notify
    messages = []
    if latest_buy_sig == 1.0:
        messages.append(f"Support Breakout (Buy) detected at {latest_index} (5m)!")
    if latest_res_sig == 1.0:
        messages.append(f"Resistance Breakout detected at {latest_index} (5m)!")

    # Send message(s) to Discord if either signal was triggered
    if messages:
        for msg in messages:
            send_discord_message(msg)

    # Optional: Print to console
    if messages:
        print("\n".join(messages))
    else:
        print(f"No breakout signals at {latest_index}.")

    # ----------------------------------
    # (Optional) Plot and save the chart
    # ----------------------------------
    # If you want to see a chart, you can plot and save it:
    # plt.style.use('dark_background')
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(data.index, data['close'], label='Close Price', color='cyan')
    # ax.plot(data.index, data['support'], label='Support', color='red')
    # ax.plot(data.index, data['resist'], label='Resistance', color='green')
    #
    # # Convert the datetime index to numeric for scatter
    # numeric_index = mdates.date2num(data.index.to_pydatetime())
    # buy_indices = data.index[data['buy_signal'] == 1.0]
    # res_break_indices = data.index[data['res_break_signal'] == 1.0]
    #
    # ax.scatter(mdates.date2num(buy_indices.to_pydatetime()),
    #            data.loc[buy_indices, 'close'],
    #            marker='^', color='lime', s=100, label='Support Breakout (Buy)')
    #
    # ax.scatter(mdates.date2num(res_break_indices.to_pydatetime()),
    #            data.loc[res_break_indices, 'close'],
    #            marker='^', color='yellow', s=100, label='Resistance Breakout')
    #
    # ax.set_title("5m ETH/USDT Trendline Breakout")
    # ax.legend()
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    # fig.autofmt_xdate()
    #
    # # Save the figure
    # plt.savefig("current_chart.png")
    # plt.close(fig)
    #
    # # You can then upload "current_chart.png" to Discord if you want an image.

# ----------------------------------
# Schedule the function to run
# ----------------------------------
schedule.every(10).minutes.do(fetch_and_check_breakouts)

# Run once immediately (so you don't wait 5 minutes for the first run)
fetch_and_check_breakouts()

print("Scheduler started. Press Ctrl+C to exit.")

# Keep the script running indefinitely
while True:
    schedule.run_pending()
    time.sleep(1)
