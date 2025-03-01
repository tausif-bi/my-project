import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from datetime import datetime, timedelta
import os
import schedule
import requests

# Discord webhook URL - Replace with your actual webhook URL
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1345284439470637079/NU2KogeP-oNzbZhonZ558NMZc2po7O58XXR7gtjM-UKXVFOEWu6Yp5EUxYIT3d0MwISz"

# Checks if there is a local top detected at curr index
def rw_top(data: np.array, curr_index: int, order: int) -> bool:
    if curr_index < order * 2 + 1:
        return False

    top = True
    k = curr_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] > v or data[k - i] > v:
            top = False
            break
    return top

# Checks if there is a local bottom detected at curr index
def rw_bottom(data: np.array, curr_index: int, order: int) -> bool:
    if curr_index < order * 2 + 1:
        return False

    bottom = True
    k = curr_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] < v or data[k - i] < v:
            bottom = False
            break
    return bottom

def rw_extremes(data: np.array, order: int):
    # Rolling window local tops and bottoms
    tops = []
    bottoms = []
    for i in range(len(data)):
        if rw_top(data, i, order):
            # top[0] = confirmation index, top[1] = index of top, top[2] = price of top
            top = [i, i - order, data[i - order]]
            tops.append(top)
        
        if rw_bottom(data, i, order):
            # bottom[0] = confirmation index, bottom[1] = index of bottom, bottom[2] = price of bottom
            bottom = [i, i - order, data[i - order]]
            bottoms.append(bottom)
    
    return tops, bottoms

def detect_line_crosses(data, resistance_levels, support_levels):
    """
    Detect when price crosses any of the support or resistance levels.
    Returns a list of crosses with timestamps, prices, and direction.
    """
    crosses = []
    
    # Combine all levels with their type
    all_levels = [(level, 'resistance') for level in resistance_levels]
    all_levels.extend([(level, 'support') for level in support_levels])
    
    # We need at least 2 data points to detect a cross
    if len(data) < 2:
        return crosses
    
    # Check each level against each candle
    for level, level_type in all_levels:
        for i in range(1, len(data)):
            prev_price = data['close'].iloc[i-1]
            curr_price = data['close'].iloc[i]
            timestamp = data.index[i]
            
            # Use both close and low/high prices for more accurate detection
            # For upward crosses, check if low/close crossed the level
            # For downward crosses, check if high/close crossed the level
            prev_low = data['low'].iloc[i-1]
            prev_high = data['high'].iloc[i-1]
            curr_low = data['low'].iloc[i]
            curr_high = data['high'].iloc[i]
            
            # Detect upward cross (price was below, now above)
            if (prev_price <= level and curr_price > level) or (prev_low <= level and curr_low > level):
                cross_type = "Bullish" if level_type == 'resistance' else "Support bounce"
                direction = "upward"
                crosses.append({
                    'timestamp': timestamp,
                    'price': curr_price,
                    'level': level,
                    'level_type': level_type,
                    'cross_type': cross_type,
                    'direction': direction,
                    'candle_range': [curr_low, curr_high]  # Store full candle range
                })
            
            # Detect downward cross (price was above, now below)
            elif (prev_price >= level and curr_price < level) or (prev_high >= level and curr_high < level):
                cross_type = "Bearish" if level_type == 'support' else "Resistance rejection"
                direction = "downward"
                crosses.append({
                    'timestamp': timestamp,
                    'price': curr_price,
                    'level': level,
                    'level_type': level_type,
                    'cross_type': cross_type,
                    'direction': direction,
                    'candle_range': [curr_low, curr_high]  # Store full candle range
                })
    
    return crosses

def live_monitor(symbol, timeframe, order, history_limit=100, update_interval=60):
    """
    Live monitor for price crosses of support and resistance levels.
    Parameters:
    - symbol: trading pair (e.g., 'BTC/USDT')
    - timeframe: candle timeframe (e.g., '5m', '1h')
    - order: order parameter for extreme detection
    - history_limit: number of historical candles to fetch
    - update_interval: seconds between updates
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Keep track of identified levels
    resistance_levels = []
    support_levels = []
    
    # Used to avoid duplicating alerts
    processed_crosses = set()
    
    print(f"Starting live monitoring of {symbol} ({timeframe}) for line crosses...")
    print(f"Press Ctrl+C to stop monitoring")
    print("-" * 80)
    
    try:
        while True:
            # Fetch latest data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=history_limit)
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
            data = data.set_index('date')
            
            # Get latest extremes
            tops, bottoms = rw_extremes(data['close'].to_numpy(), order)
            
            # Extract price levels
            current_resistance_levels = [top[2] for top in tops]
            current_support_levels = [bottom[2] for bottom in bottoms]
            
            # Update our tracked levels
            resistance_levels = current_resistance_levels
            support_levels = current_support_levels
            
            # Detect crosses
            crosses = detect_line_crosses(data, resistance_levels, support_levels)
            
            # Only alert for new crosses in the most recent candle
            latest_time = data.index[-1]
            recent_time_threshold = latest_time - timedelta(minutes=int(timeframe[:-1]) * 2)  # 2 candles ago
            
            for cross in crosses:
                # Create a unique identifier for this cross
                cross_id = f"{cross['timestamp']}_{cross['level']}_{cross['direction']}"
                
                # Only process if this is a new cross and it's recent
                if cross_id not in processed_crosses and cross['timestamp'] >= recent_time_threshold:
                    print(f"[{cross['timestamp']}] {cross['cross_type']} cross detected!")
                    print(f"  Price: {cross['price']:.2f} crossed {cross['level_type']} level at {cross['level']:.2f}")
                    print(f"  Direction: {cross['direction']}")
                    print("-" * 80)
                    
                    # Mark as processed
                    processed_crosses.add(cross_id)
            
            # Sleep until next update
            print(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Next update in {update_interval} seconds")
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

def run_analysis(symbol='BTC/USDT', timeframe='5m', order=4, limit=500, show_plot=True, monitor_live=False, send_to_discord=False, zoom_level=20):
    """
    Run the complete analysis flow - fetch data, identify extremes, plot, and optionally monitor.
    
    Parameters:
    - symbol: Trading pair (e.g., 'BTC/USDT')
    - timeframe: Candle timeframe (e.g., '5m', '1h')
    - order: Order parameter for extreme detection
    - limit: Number of candles to fetch for analysis
    - show_plot: Whether to display the plot on screen
    - monitor_live: Whether to start live monitoring after analysis
    - send_to_discord: Whether to send the chart to Discord
    - zoom_level: Number of candles to show in the chart for Discord (must be <= limit)
    """
    # Instead of reading from CSV, we use ccxt to fetch data from Binance.
    exchange = ccxt.binance({'enableRateLimit': True})
    
    print(f"Fetching {limit} {timeframe} candles for {symbol}...")
    # Fetch OHLCV data: [timestamp, open, high, low, close, volume]
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    
    # Create DataFrame with appropriate column names
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data.set_index('date')
    
    # Compute rolling window extremes on the close prices
    tops, bottoms = rw_extremes(data['close'].to_numpy(), order)
    
    # Print tops with their dates
    print("== Local Tops ==")
    for top in tops:
        print(f"Top index: {top[1]}, Date: {data.index[top[1]]}, Price: {top[2]}")

    print("\n== Local Bottoms ==")
    for bottom in bottoms:
        print(f"Bottom index: {bottom[1]}, Date: {data.index[bottom[1]]}, Price: {bottom[2]}")

    print(f"\nData start: {data.index[0]}")
    print(f"Data end: {data.index[-1]}")
    
    # Extract just the prices for resistance and support levels
    resistance_levels = [top[2] for top in tops]
    support_levels = [bottom[2] for bottom in bottoms]
    
    # Check for recent line crosses
    recent_crosses = detect_line_crosses(data, resistance_levels, support_levels)
    
    # Show the 10 most recent crosses
    cross_text = ""
    if recent_crosses:
        print("\n== Recent Line Crosses ==")
        for cross in sorted(recent_crosses, key=lambda x: x['timestamp'], reverse=True)[:10]:
            line_text = f"[{cross['timestamp']}] {cross['cross_type']} - Price: {cross['price']:.2f} crossed {cross['level_type']} at {cross['level']:.2f} ({cross['direction']})"
            print(line_text)
            cross_text += line_text + "\n"
    
    # Create chart figure
    plt.figure(figsize=(14, 8))
    
    # Determine what portion of data to show in the plot
    # For Discord, show only the last zoom_level candles; for local display, show all
    display_data = data
    display_start = data.index[0]
    if send_to_discord and not show_plot and zoom_level < len(data):
        display_data = data.iloc[-zoom_level:]
        display_start = display_data.index[0]
        print(f"Chart will be zoomed to show the last {zoom_level} candles")
    
    # Plot the close prices
    plt.plot(display_data.index, display_data['close'], linewidth=1)
    
    # Enhanced marker size and visibility
    marker_size = 100  # Increase marker size
    alpha_value = 0.7  # Semi-transparent markers
    
    # Get the right edge of the plot for extending rays
    right_edge = display_data.index[-1]
    
    # For Discord plots that are zoomed in, we still want to show ALL support/resistance lines
    # even if the marker that created them is outside the visible range
    if send_to_discord and not show_plot and zoom_level < len(data):
        # Plot ALL tops' horizontal rays, even those outside the visible range
        for top in tops:
            # Only plot the horizontal line portion (not the marker)
            plt.hlines(y=top[2], xmin=display_start, xmax=right_edge, 
                      colors='green', linestyles='dashed', alpha=0.4, linewidth=1)
        
        # Plot ALL bottoms' horizontal rays, even those outside the visible range
        for bottom in bottoms:
            # Only plot the horizontal line portion (not the marker)
            plt.hlines(y=bottom[2], xmin=display_start, xmax=right_edge, 
                      colors='red', linestyles='dashed', alpha=0.4, linewidth=1)
    
    # Filter tops and bottoms to only show markers within the display range
    display_tops = [top for top in tops if data.index[top[1]] >= display_start]
    display_bottoms = [bottom for bottom in bottoms if data.index[bottom[1]] >= display_start]
    
    # Plot top markers (only within visible range)
    for top in display_tops:
        if data.index[top[1]] in display_data.index:
            # Plot the marker
            plt.scatter(data.index[top[1]], top[2], 
                      s=marker_size, marker='^', color='green', 
                      alpha=alpha_value, zorder=5)
            
            # For non-zoomed display, plot the horizontal ray
            if not (send_to_discord and not show_plot and zoom_level < len(data)):
                plt.hlines(y=top[2], xmin=data.index[top[1]], xmax=right_edge, 
                          colors='green', linestyles='dashed', alpha=0.4, linewidth=1)
    
    # Plot bottom markers (only within visible range)
    for bottom in display_bottoms:
        if data.index[bottom[1]] in display_data.index:
            # Plot the marker
            plt.scatter(data.index[bottom[1]], bottom[2], 
                      s=marker_size, marker='v', color='red', 
                      alpha=alpha_value, zorder=5)
            
            # For non-zoomed display, plot the horizontal ray
            if not (send_to_discord and not show_plot and zoom_level < len(data)):
                plt.hlines(y=bottom[2], xmin=data.index[bottom[1]], xmax=right_edge, 
                          colors='red', linestyles='dashed', alpha=0.4, linewidth=1)
    
    # Filter crosses to only show those within the display range
    display_crosses = [cross for cross in recent_crosses if cross['timestamp'] >= display_start]
    
    # Add markers for recent crosses
    for cross in display_crosses:
        if cross['direction'] == "upward":
            plt.scatter(cross['timestamp'], cross['price'], 
                      s=80, marker='o', color='blue', edgecolors='white', 
                      linewidth=2, zorder=6)
            # Add annotation with the exact price
            plt.annotate(f"{cross['price']:.2f}", 
                       (cross['timestamp'], cross['price']),
                       xytext=(0, 10), textcoords='offset points',
                       fontsize=8, color='blue')
        else:  # downward
            plt.scatter(cross['timestamp'], cross['price'], 
                      s=80, marker='o', color='purple', edgecolors='white', 
                      linewidth=2, zorder=6)
            # Add annotation with the exact price
            plt.annotate(f"{cross['price']:.2f}", 
                       (cross['timestamp'], cross['price']),
                       xytext=(0, -15), textcoords='offset points',
                       fontsize=8, color='purple')
            
    # Improve plot formatting
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    chart_title = f'{symbol} {timeframe} Price with Local Extremes (Order={order}) - {current_time}'
    if send_to_discord and zoom_level < limit:
        chart_title += f' (Showing last {zoom_level} of {limit} candles)'
    plt.title(chart_title)
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    
    # For Discord zoomed charts, adjust y-axis to focus on visible price range
    if send_to_discord and not show_plot and zoom_level < len(data):
        # Get price range of visible data with some padding
        y_min = display_data['close'].min()
        y_max = display_data['close'].max()
        
        # Add padding (5% of the range)
        padding = (y_max - y_min) * 0.15
        plt.ylim(y_min - padding, y_max + padding)
        
        # Add text annotations for important support/resistance levels outside the visible range
        visible_price_range = plt.gca().get_ylim()
        
        # Filter levels that are outside the visible range but close enough to be relevant
        off_chart_levels = []
        for level in resistance_levels:
            if level < visible_price_range[0] and level > visible_price_range[0] - (y_max - y_min):
                off_chart_levels.append(("R", level))
        for level in support_levels:
            if level < visible_price_range[0] and level > visible_price_range[0] - (y_max - y_min):
                off_chart_levels.append(("S", level))
                
        if off_chart_levels:
            off_levels_text = "Notable levels below chart: " + ", ".join([f"{t}:{v:.2f}" for t, v in off_chart_levels])
            plt.figtext(0.5, 0.01, off_levels_text, ha='center', fontsize=8)
    
    # Add a legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='Local Top'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, label='Local Bottom'),
        Line2D([0], [0], color='green', linestyle='dashed', alpha=0.4, label='Resistance Level'),
        Line2D([0], [0], color='red', linestyle='dashed', alpha=0.4, label='Support Level'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Upward Cross'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Downward Cross')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    # Save the figure if sending to Discord
    if send_to_discord:
        # Create directory for images if it doesn't exist
        os.makedirs("chart_images", exist_ok=True)
        
        # Save to file
        image_path = f"chart_images/{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(image_path, dpi=120)
        print(f"Chart saved as {image_path}")
        
        # Send to Discord
        send_chart_to_discord(image_path, cross_text, symbol, timeframe)
        
        # Delete the file after sending
        os.remove(image_path)
        print(f"Deleted image: {image_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # If monitoring mode is enabled, start live monitoring
    if monitor_live:
        live_monitor(symbol, timeframe, order)

def send_chart_to_discord(image_path, cross_text, symbol, timeframe):
    """Send the chart image to Discord webhook"""
    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create message content
        message_content = f"**{symbol} {timeframe} Analysis** - *{current_time}*"
        if cross_text:
            message_content += f"\n\n```\nRecent Line Crosses:\n{cross_text}```"
        
        # Prepare the file
        files = {
            'file': (os.path.basename(image_path), open(image_path, 'rb'), 'image/png')
        }
        
        # Prepare the payload
        payload = {
            'content': message_content
        }
        
        # Send to Discord
        response = requests.post(
            DISCORD_WEBHOOK_URL,
            data=payload,
            files=files
        )
        
        # Check response
        if response.status_code == 204:
            print("Successfully sent to Discord!")
        else:
            print(f"Failed to send to Discord: Status code {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error sending to Discord: {e}")

def scheduled_task():
    """Function to run on schedule"""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running scheduled analysis...")
    run_analysis(
        symbol='BTC/USDT',
        timeframe='5m',
        order=4,
        limit=500,
        show_plot=False,
        monitor_live=False,
        send_to_discord=True,
        zoom_level=20
    )
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Scheduled analysis complete.")

def run_scheduler():
    """Start the scheduler to run analysis every 5 minutes"""
    print(f"Starting automatic chart generation service - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Charts will be generated and sent to Discord every 5 minutes.")
    print("Press Ctrl+C to stop the service.")
    print("-" * 80)
    
    # Run once at startup
    scheduled_task()
    
    # Schedule to run every 5 minutes
    schedule.every(60).minutes.do(scheduled_task)
    
    # Keep the scheduler running
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nScheduler stopped by user.")

if __name__ == "__main__":
    # For a one-time analysis with plot display:
    # Uncomment this to show the chart once immediately on your local machine
    # run_analysis(symbol='BTC/USDT', timeframe='5m', order=1, limit=100, show_plot=True)
    
    # Then start the scheduler for continuous 5-minute runs
    run_scheduler()