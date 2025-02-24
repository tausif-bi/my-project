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
import os

# ----------------------------------
# Discord Webhook Configuration
# ----------------------------------
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1342868833245073448/JcPbZr57wAOneJrKUO0PueBRI0my0SGu7YoIpyhs-5rtRsgTAO6_IcHLR45VqNu9TW8O"

def send_discord_message(message: str):
    """Send a text message to Discord via webhook."""
    payload = {"content": message}
    requests.post(DISCORD_WEBHOOK_URL, json=payload)

def send_discord_image(file_path: str, message: str = ""):
    """Send an image file to Discord via webhook."""
    url = DISCORD_WEBHOOK_URL
    data = {"content": message}
    with open(file_path, "rb") as file:
        files = {"file": (file_path, file)}
        response = requests.post(url, data=data, files=files)
    if response.status_code == 204:
        print("Image sent successfully!")
    else:
        print(f"Failed to send image. Status code: {response.status_code}")

# ----------------------------------
# Trendline breakout function
# ----------------------------------
def trendline_breakout(close: np.array, lookback: int):
    # Initialize arrays for support and resistance trendlines
    s_tl = np.full(len(close), np.nan)
    r_tl = np.full(len(close), np.nan)

    # Initialize signals: buy_sig for breakout above support,
    # and res_break_sig for breakout above resistance.
    buy_sig = np.zeros(len(close))
    res_break_sig = np.zeros(len(close))

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

        # 1) Crossing Above Support: Check if the previous close was at or below support and current is above.
        if i > lookback and (close[i - 1] <= s_tl[i - 1]) and (close[i] > s_val):
            buy_sig[i] = 1.0  # Buy signal triggered

        # 2) Crossing Above Resistance: Check if the previous close was at or below resistance and current is above.
        if i > lookback and (close[i - 1] <= r_tl[i - 1]) and (close[i] > r_val):
            res_break_sig[i] = 1.0

    return s_tl, r_tl, buy_sig, res_break_sig

# ----------------------------------
# List of symbols to monitor
# ----------------------------------
symbols = [
    'BTC/USDT', 'XRP/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT',
    'TRX/USDT', 'LINK/USDT', 'SUI/USDT', 'AVAX/USDT', 'XLM/USDT',
    'TON/USDT', 'HBAR/USDT', 'DOT/USDT'
]

def fetch_and_check_breakouts():
    """
    Loop over the list of symbols, fetch data for each, compute trendlines,
    detect breakouts on the most recent candle, send Discord notifications,
    plot & save the chart image, and then delete the image.
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    timeframe = '1h'
    limit = 100
    lookback = 72

    for symbol in symbols:
        try:
            # Fetch OHLCV data: [timestamp, open, high, low, close, volume]
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
            data = data.set_index('date').astype(float)
            data = data.dropna()

            # Run the breakout logic
            support, resist, buy_signal, res_break_signal = trendline_breakout(data['close'].to_numpy(), lookback)
            data['support'] = support
            data['resist'] = resist
            data['buy_signal'] = buy_signal
            data['res_break_signal'] = res_break_signal

            # Check the most recent candle for breakout signals
            latest_index = data.index[-1]
            latest_buy_sig = data['buy_signal'].iloc[-1]
            latest_res_sig = data['res_break_signal'].iloc[-1]

            messages = []
            if latest_buy_sig == 1.0:
                messages.append(f"{symbol}: Support Breakout (Buy) detected at {latest_index} (1h)!")
            if latest_res_sig == 1.0:
                messages.append(f"{symbol}: Resistance Breakout detected at {latest_index} (1h)!")

            # Send text notifications to Discord if any signal is triggered
            if messages:
                for msg in messages:
                    send_discord_message(msg)
                    print(msg)
            else:
                print(f"{symbol}: No breakout signals at {latest_index}.")

            # ----------------------------------
            # Plot and save the chart
            # ----------------------------------
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data['close'], label='Close Price', color='cyan')
            ax.plot(data.index, data['support'], label='Support', color='red')
            ax.plot(data.index, data['resist'], label='Resistance', color='green')

            # Convert datetime index to numeric for scatter plotting
            numeric_index = mdates.date2num(data.index.to_pydatetime())
            buy_indices = data.index[data['buy_signal'] == 1.0]
            res_break_indices = data.index[data['res_break_signal'] == 1.0]

            ax.scatter(mdates.date2num(buy_indices.to_pydatetime()),
                       data.loc[buy_indices, 'close'],
                       marker='^', color='lime', s=100, label='Support Breakout (Buy)')

            ax.scatter(mdates.date2num(res_break_indices.to_pydatetime()),
                       data.loc[res_break_indices, 'close'],
                       marker='^', color='yellow', s=100, label='Resistance Breakout')

            ax.set_title(f"{symbol} Trendline Breakout")
            ax.legend()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            fig.autofmt_xdate()

            chart_filename = f"{symbol.replace('/', '_')}_chart.png"
            plt.savefig(chart_filename)
            plt.close(fig)

            # Send the chart image if any breakout signal was triggered, then delete the image
            if messages:
                send_discord_image(chart_filename, message=f"Chart for {symbol} Trendline Breakout")
                os.remove(chart_filename)

            # Pause briefly to avoid hitting rate limits
            time.sleep(1)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

# ----------------------------------
# Schedule the function to run every 10 minutes
# ----------------------------------
schedule.every(10).minutes.do(fetch_and_check_breakouts)

# Run once immediately (so you don't wait for the scheduler)
fetch_and_check_breakouts()

print("Scheduler started. Press Ctrl+C to exit.")

# Keep the script running indefinitely
while True:
    schedule.run_pending()
    time.sleep(1)
