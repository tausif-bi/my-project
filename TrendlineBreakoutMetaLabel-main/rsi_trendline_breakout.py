import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trendline_automation import fit_trendlines_single
import matplotlib.dates as mdates
import schedule
import time
import requests
import os
from datetime import datetime
from matplotlib.gridspec import GridSpec
from db_utils import store_signal, parse_signal_timestamp

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
    if response.status_code in [200, 204]:  # Accept both 200 and 204 as success
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
    'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT',
    'TRX/USDT', 'LINK/USDT', 'SUI/USDT', 'AVAX/USDT', 'XLM/USDT', 'DOT/USDT'
]

def create_dashboard():
    """
    Create a dashboard showing RSI-10 trendline breakouts for all symbols.
    """
    # Define parameters
    exchange = ccxt.binance({'enableRateLimit': True})
    timeframe = '15m'  # 1-minute timeframe for testing
    limit = 100  # Fetch 100 candles
    lookback = 30  # Lookback for trendline detection
    rsi_period = 10  # RSI-10
    
    # Create figure for dashboard
    num_symbols = len(symbols)
    rows = (num_symbols + 1) // 2  # Calculate rows needed (2 symbols per row)
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, rows * 5))
    gs = GridSpec(rows, 2, figure=fig)
    
    # Store signals for summary
    active_signals = []
    dashboard_file = "rsi_trendline_dashboard.png"
    
    # Process each symbol
    for i, symbol in enumerate(symbols):
        try:
            print(f"Processing {symbol} for dashboard...")
            
            # Fetch data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
            data = data.set_index('date').astype(float)
            
            # Calculate RSI
            data['rsi'] = calculate_rsi(data['close'], period=rsi_period)
            data = data.dropna()  # Remove NaN values
            
            # Run breakout detection
            support, resist, buy_signal, res_break_signal = rsi_trendline_breakout(data['rsi'].to_numpy(), lookback)
            data['rsi_support'] = support
            data['rsi_resist'] = resist
            data['buy_signal'] = buy_signal
            data['res_break_signal'] = res_break_signal
            
            # Check last 3 candles for signals (to catch recent signals)
            recent_buy_sig = data['buy_signal'].iloc[-3:].any()
            recent_res_sig = data['res_break_signal'].iloc[-3:].any()
            
            # Current values
            current_rsi = data['rsi'].iloc[-1]
            current_price = data['close'].iloc[-1]
            current_time = data.index[-1]
            
            # Calculate row and column for subplot
            row = i // 2
            col = i % 2
            
            # Create subplot
            ax = fig.add_subplot(gs[row, col])
            
            # Plot RSI
            ax.plot(data.index[-30:], data['rsi'].iloc[-30:], color='magenta', linewidth=2)
            ax.plot(data.index[-30:], data['rsi_support'].iloc[-30:], 'r--', alpha=0.5)
            ax.plot(data.index[-30:], data['rsi_resist'].iloc[-30:], 'g--', alpha=0.5)
            
            # Add RSI reference lines
            ax.axhline(y=70, color='r', linestyle=':', alpha=0.3)
            ax.axhline(y=30, color='g', linestyle=':', alpha=0.3)
            ax.axhline(y=50, color='y', linestyle=':', alpha=0.3)
            
            # Add signals to plot
            buy_indices = data.index[data['buy_signal'] == 1.0]
            res_indices = data.index[data['res_break_signal'] == 1.0]
            
            # Only plot recent signals (last 30 candles)
            recent_buy = [idx for idx in buy_indices if idx in data.index[-30:]]
            recent_res = [idx for idx in res_indices if idx in data.index[-30:]]
            
            if recent_buy:
                ax.scatter(recent_buy, data.loc[recent_buy, 'rsi'], 
                          marker='^', color='lime', s=100)
            
            if recent_res:
                ax.scatter(recent_res, data.loc[recent_res, 'rsi'], 
                          marker='^', color='yellow', s=100)
            
            # Title styling based on signals
            title_color = 'white'
            if recent_buy_sig:
                title_color = 'lime'
                signal_msg = f"{symbol}: RSI-10 Support Breakout (Buy) detected at {current_time} ({timeframe})! RSI: {current_rsi:.2f}, Price: {current_price:.2f}"
                active_signals.append(f"{symbol}: Support Breakout - RSI:{current_rsi:.1f}")

                # Store in database
                store_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_time=current_time,
                    price=current_price,
                    signal_type='rsi_breakout',
                    signal_details=signal_msg,
                    direction='upward',
                    level_type='support',
                    cross_type='RSI Support Breakout',
                    rsi_value=current_rsi,
                    chart_image_path=dashboard_file if dashboard_file else None
                )

            elif recent_res_sig:
                title_color = 'yellow'
                signal_msg = f"{symbol}: RSI-10 Resistance Breakout detected at {current_time} ({timeframe})! RSI: {current_rsi:.2f}, Price: {current_price:.2f}"
                active_signals.append(f"{symbol}: Resistance Breakout - RSI:{current_rsi:.1f}")

                # Store in database
                store_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_time=current_time,
                    price=current_price,
                    signal_type='rsi_breakout',
                    signal_details=signal_msg,
                    direction='upward',
                    level_type='resistance',
                    cross_type='RSI Resistance Breakout',
                    rsi_value=current_rsi,
                    chart_image_path=dashboard_file if dashboard_file else None
                )
                        
            # Set title and limits
            ax.set_title(f"{symbol} - RSI:{current_rsi:.1f} - ${current_price:.2f}", color=title_color, fontweight='bold')
            ax.set_ylim(0, 100)
            ax.set_ylabel('RSI-10')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.2)
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            # Create empty subplot with error message
            row = i // 2
            col = i % 2
            ax = fig.add_subplot(gs[row, col])
            ax.text(0.5, 0.5, f"{symbol}: Error - {str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='red')
            ax.set_axis_off()
    
    # Add dashboard title with timestamp and summary
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    title = f"RSI-10 Trendline Breakout Dashboard - {current_time}\n"
    
    if active_signals:
        title += "Active Signals: " + " | ".join(active_signals)
    else:
        title += "No active signals"
        
    fig.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save dashboard
    dashboard_file = "rsi_trendline_dashboard.png"
    plt.savefig(dashboard_file, dpi=120)
    plt.close(fig)
    
    # Send to Discord with summary message
    message = f"RSI-10 Trendline Breakout Dashboard Update - {current_time}"
    if active_signals:
        message += "\n**ACTIVE SIGNALS:**"
        for signal in active_signals:
            message += f"\n {signal}"
    
    send_discord_image(dashboard_file, message=message)
    
    # Delete file after sending
    os.remove(dashboard_file)
    print(f"Dashboard updated at {current_time}")
    
    return active_signals

# ----------------------------------
# Schedule dashboard updates
# ----------------------------------
def run_scheduler():
    # First run immediately
    print("Creating initial dashboard...")
    create_dashboard()
    
    # Schedule to run every 5 minutes
    # For testing with 1m timeframe, 5 minutes is a good balance
    schedule.every(15).minutes.do(create_dashboard)
    
    print("Dashboard scheduler started. Will update every 5 minutes.")
    print("Press Ctrl+C to exit.")
    
    # Keep the script running
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDashboard updates stopped.")

if __name__ == "__main__":
    run_scheduler()