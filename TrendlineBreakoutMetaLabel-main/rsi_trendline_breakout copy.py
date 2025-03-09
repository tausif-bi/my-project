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
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1345348754315673652/z4DMmZZjQzchCcInAyIDYH3N7NqN6rKb7jfRX0-f7b_KOg9ofIW-nlNpF3OIalCK-zU8"

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
# Calculate RSI (Relative Strength Index)
# ----------------------------------
def calculate_rsi(prices, period=10):
    """Calculate the RSI for a given price series."""
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# ----------------------------------
# Trendline breakout function for RSI
# ----------------------------------
def rsi_trendline_breakout(rsi_values: np.array, lookback: int):
    # Initialize arrays for support and resistance trendlines
    s_tl = np.full(len(rsi_values), np.nan)
    r_tl = np.full(len(rsi_values), np.nan)

    # Initialize signals: buy_sig for breakout above support,
    # and res_break_sig for breakout above resistance.
    buy_sig = np.zeros(len(rsi_values))
    res_break_sig = np.zeros(len(rsi_values))

    for i in range(lookback, len(rsi_values)):
        # Use the past 'lookback' candles to compute trendlines
        window = rsi_values[i - lookback: i]

        # Calculate support and resistance trendline coefficients
        s_coefs, r_coefs = fit_trendlines_single(window)

        # Project the trendlines to the current index
        s_val = s_coefs[1] + lookback * s_coefs[0]
        r_val = r_coefs[1] + lookback * r_coefs[0]

        s_tl[i] = s_val
        r_tl[i] = r_val

        # 1) Crossing Above Support: Check if the previous RSI was at or below support and current is above.
        if i > lookback and (rsi_values[i - 1] <= s_tl[i - 1]) and (rsi_values[i] > s_val):
            buy_sig[i] = 1.0  # Buy signal triggered

        # 2) Crossing Above Resistance: Check if the previous RSI was at or below resistance and current is above.
        if i > lookback and (rsi_values[i - 1] <= r_tl[i - 1]) and (rsi_values[i] > r_val):
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
    Loop over the list of symbols, fetch data for each, calculate RSI, compute trendlines on RSI,
    detect breakouts on the most recent candle, send Discord notifications,
    plot & save the chart image, and then delete the image.
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    timeframe = '1m'
    limit = 100  # Increased limit to have enough data for RSI calculation
    lookback = 72
    rsi_period = 10  # RSI-10 as requested

    for symbol in symbols:
        try:
            print(f"Processing {symbol}...")
            
            # Fetch OHLCV data: [timestamp, open, high, low, close, volume]
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
            data = data.set_index('date').astype(float)
            data = data.dropna()

            # Calculate RSI-10
            data['rsi'] = calculate_rsi(data['close'], period=rsi_period)
            
            # Drop NaN values (first few RSI values will be NaN)
            data = data.dropna()

            # Run the breakout logic on RSI values
            support, resist, buy_signal, res_break_signal = rsi_trendline_breakout(data['rsi'].to_numpy(), lookback)
            data['rsi_support'] = support
            data['rsi_resist'] = resist
            data['buy_signal'] = buy_signal
            data['res_break_signal'] = res_break_signal

            # Check the most recent candle for breakout signals
            latest_index = data.index[-1]
            latest_buy_sig = data['buy_signal'].iloc[-1]
            latest_res_sig = data['res_break_signal'].iloc[-1]
            latest_rsi = data['rsi'].iloc[-1]
            latest_price = data['close'].iloc[-1]

            messages = []
            if latest_buy_sig == 1.0:
                messages.append(f"{symbol}: RSI-10 Support Breakout (Buy) detected at {latest_index} (1h)! RSI: {latest_rsi:.2f}, Price: {latest_price:.2f}")
            if latest_res_sig == 1.0:
                messages.append(f"{symbol}: RSI-10 Resistance Breakout detected at {latest_index} (1h)! RSI: {latest_rsi:.2f}, Price: {latest_price:.2f}")

            # Send text notifications to Discord if any signal is triggered
            if messages:
                for msg in messages:
                    send_discord_message(msg)
                    print(msg)
            else:
                print(f"{symbol}: No RSI breakout signals at {latest_index}.")

            # ----------------------------------
            # Plot and save the chart
            # ----------------------------------
            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
            
            # Plot price on top subplot
            ax1.plot(data.index, data['close'], label='Close Price', color='cyan')
            ax1.set_title(f"{symbol} Price and RSI-10 Trendline Breakout")
            ax1.legend(loc='upper left')
            ax1.set_ylabel('Price')
            
            # Plot RSI and trendlines on bottom subplot
            ax2.plot(data.index, data['rsi'], label='RSI-10', color='magenta')
            ax2.plot(data.index, data['rsi_support'], label='RSI Support', color='red')
            ax2.plot(data.index, data['rsi_resist'], label='RSI Resistance', color='green')
            
            # Add RSI reference lines
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.3)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.3)
            ax2.axhline(y=50, color='y', linestyle='--', alpha=0.3)
            
            # Plot breakout signals on RSI chart
            # Convert datetime index to numeric for scatter plotting
            buy_indices = data.index[data['buy_signal'] == 1.0]
            res_break_indices = data.index[data['res_break_signal'] == 1.0]

            ax2.scatter(mdates.date2num(buy_indices.to_pydatetime()),
                       data.loc[buy_indices, 'rsi'],
                       marker='^', color='lime', s=100, label='RSI Support Breakout')

            ax2.scatter(mdates.date2num(res_break_indices.to_pydatetime()),
                       data.loc[res_break_indices, 'rsi'],
                       marker='^', color='yellow', s=100, label='RSI Resistance Breakout')
                       
            ax2.set_ylabel('RSI-10')
            ax2.set_ylim(0, 100)  # RSI ranges from 0 to 100
            ax2.legend(loc='upper left')
            
            # Format x-axis dates
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                ax.grid(True, alpha=0.2)
            
            fig.autofmt_xdate()
            plt.tight_layout()

            chart_filename = f"{symbol.replace('/', '_')}_rsi_chart.png"
            plt.savefig(chart_filename)
            plt.close(fig)

            # Send the chart image if any breakout signal was triggered, then delete the image
            if messages:
                send_discord_image(chart_filename, message=f"Chart for {symbol} RSI-10 Trendline Breakout")
                os.remove(chart_filename)
            elif latest_index.hour % 6 == 0:  # Send regular updates every 6 hours even if no signal
                send_discord_image(chart_filename, message=f"Regular update for {symbol} - RSI-10: {latest_rsi:.2f}")
                os.remove(chart_filename)
            else:
                os.remove(chart_filename)  # Delete the chart if not sent

            # Pause briefly to avoid hitting rate limits
            time.sleep(1)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

# ----------------------------------
# Schedule the function to run every 10 minutes
# ----------------------------------
schedule.every(1).minutes.do(fetch_and_check_breakouts)

# Run once immediately (so you don't wait for the scheduler)
fetch_and_check_breakouts()

print("Scheduler started. Monitoring RSI-10 trendline breakouts. Press Ctrl+C to exit.")

# Keep the script running indefinitely
try:
    while True:
        schedule.run_pending()
        time.sleep(1)
except KeyboardInterrupt:
    print("\nMonitoring stopped.")