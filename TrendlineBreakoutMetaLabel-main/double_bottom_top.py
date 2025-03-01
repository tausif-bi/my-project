# Double Bottom/Top Pattern Detector using CCXT and Binance
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as patches

def fetch_ohlcv_data(symbol='BTC/USDT', timeframe='5m', limit=200):
    """
    Fetch OHLCV data from Binance using CCXT
    """
    print(f"Fetching {limit} {timeframe} candles for {symbol} from Binance...")
    
    # Initialize Binance exchange
    exchange = ccxt.binance()
    
    try:
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"Successfully fetched {len(df)} candles.")
        return df
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def find_patterns(df, window_size=15, threshold=0.02, significance_pct=0.5):
    """
    Find double bottom and double top patterns in the price data
    
    Parameters:
    - df: DataFrame with OHLCV data
    - window_size: Number of candles to consider for pattern formation
    - threshold: Price difference threshold for considering bottoms/tops equal (e.g., 0.02 = 2%)
    - significance_pct: Minimum percentage move required for a pattern to be significant (0.5% default)
    
    Returns:
    - Dictionary with double bottom and double top patterns
    """
    patterns = {
        'double_bottoms': [],
        'double_tops': []
    }
    
    # Calculate price range and minimum swing required for significance
    price_range = df['high'].max() - df['low'].min()
    min_swing = price_range * (significance_pct / 100)
    
    # Find stronger local minima and maxima (more significant swings)
    df['is_min'] = False
    df['is_max'] = False
    
    # Look for more significant swing lows (bottoms)
    for i in range(2, len(df) - 2):
        # Check if this is a local minimum
        if (df.iloc[i]['low'] < df.iloc[i-1]['low'] and 
            df.iloc[i]['low'] < df.iloc[i-2]['low'] and
            df.iloc[i]['low'] < df.iloc[i+1]['low'] and 
            df.iloc[i]['low'] < df.iloc[i+2]['low']):
            
            # Check if the swing is significant enough
            left_swing = df.iloc[i-2:i]['high'].max() - df.iloc[i]['low']
            right_swing = df.iloc[i+1:i+3]['high'].max() - df.iloc[i]['low']
            
            if left_swing >= min_swing and right_swing >= min_swing:
                df.at[df.index[i], 'is_min'] = True
    
    # Look for more significant swing highs (tops)
    for i in range(2, len(df) - 2):
        # Check if this is a local maximum
        if (df.iloc[i]['high'] > df.iloc[i-1]['high'] and 
            df.iloc[i]['high'] > df.iloc[i-2]['high'] and
            df.iloc[i]['high'] > df.iloc[i+1]['high'] and 
            df.iloc[i]['high'] > df.iloc[i+2]['high']):
            
            # Check if the swing is significant enough
            left_swing = df.iloc[i]['high'] - df.iloc[i-2:i]['low'].min()
            right_swing = df.iloc[i]['high'] - df.iloc[i+1:i+3]['low'].min()
            
            if left_swing >= min_swing and right_swing >= min_swing:
                df.at[df.index[i], 'is_max'] = True
    
    # Get indices of minima and maxima
    minima = df[df['is_min']].copy()
    maxima = df[df['is_max']].copy()
    
    print(f"Found {len(minima)} significant bottoms and {len(maxima)} significant tops")
    
    # Use a smarter approach to find fewer, more meaningful double bottoms
    # Sort minima by their significance (depth of the bottom)
    minima_list = []
    for idx, row in minima.iterrows():
        i = df.index.get_loc(idx)
        # Look 5 candles before and after for the highest point
        range_start = max(0, i - 5)
        range_end = min(len(df) - 1, i + 5)
        highest_around = df.iloc[range_start:range_end+1]['high'].max()
        depth = highest_around - row['low']
        minima_list.append((i, row['low'], depth))
    
    # Sort by depth (most significant first)
    minima_list.sort(key=lambda x: x[2], reverse=True)
    
    # Take only the top 50% most significant bottoms
    if len(minima_list) > 4:
        minima_list = minima_list[:len(minima_list)//2]
    
    # Find double bottoms from the significant minima
    for idx1, min1_val, _ in minima_list:
        for idx2, min2_val, _ in minima_list:
            if idx1 >= idx2 or idx2 - idx1 > window_size:
                continue  # Skip if not in sequence or too far apart
            
            # Check if bottoms are approximately equal
            price_diff = abs(min2_val - min1_val) / min1_val
            
            if price_diff <= threshold:
                # Check if there's a significant peak between the two bottoms
                between_maxima = [max_idx for max_idx, _, _ in 
                                 [(df.index.get_loc(idx), row['high'], 0) for idx, row in maxima.iterrows()]
                                 if max_idx > idx1 and max_idx < idx2]
                
                if between_maxima:
                    # Find the highest high between bottoms for confirmation
                    mid_idx = min(between_maxima, key=lambda x: abs((idx1 + idx2) / 2 - x))
                    max_between = df.iloc[mid_idx]['high']
                    
                    # Check if price broke above this level after the second bottom
                    breakout_idx = min(idx2 + 15, len(df) - 1)
                    breakout_confirmed = any(df.iloc[idx2+1:breakout_idx+1]['close'] > max_between)
                    
                    # Add to patterns if not too similar to existing ones
                    is_duplicate = False
                    for pattern in patterns['double_bottoms']:
                        existing_idx1, existing_val1 = pattern['first']
                        existing_idx2, existing_val2 = pattern['second']
                        
                        # Check if new pattern overlaps with existing one
                        if (abs(idx1 - existing_idx1) <= 3 or 
                            abs(idx2 - existing_idx2) <= 3):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        patterns['double_bottoms'].append({
                            'first': (idx1, min1_val),
                            'second': (idx2, min2_val),
                            'confirmed': breakout_confirmed,
                            'confirmation_price': max_between
                        })
    
    # Similar approach for double tops
    # Sort maxima by their significance (height of the top)
    maxima_list = []
    for idx, row in maxima.iterrows():
        i = df.index.get_loc(idx)
        # Look 5 candles before and after for the lowest point
        range_start = max(0, i - 5)
        range_end = min(len(df) - 1, i + 5)
        lowest_around = df.iloc[range_start:range_end+1]['low'].min()
        height = row['high'] - lowest_around
        maxima_list.append((i, row['high'], height))
    
    # Sort by height (most significant first)
    maxima_list.sort(key=lambda x: x[2], reverse=True)
    
    # Take only the top 50% most significant tops
    if len(maxima_list) > 4:
        maxima_list = maxima_list[:len(maxima_list)//2]
    
    # Find double tops from the significant maxima
    for idx1, max1_val, _ in maxima_list:
        for idx2, max2_val, _ in maxima_list:
            if idx1 >= idx2 or idx2 - idx1 > window_size:
                continue  # Skip if not in sequence or too far apart
            
            # Check if tops are approximately equal
            price_diff = abs(max2_val - max1_val) / max1_val
            
            if price_diff <= threshold:
                # Check if there's a significant trough between the two tops
                between_minima = [min_idx for min_idx, _, _ in 
                                 [(df.index.get_loc(idx), row['low'], 0) for idx, row in minima.iterrows()]
                                 if min_idx > idx1 and min_idx < idx2]
                
                if between_minima:
                    # Find the lowest low between tops for confirmation
                    mid_idx = min(between_minima, key=lambda x: abs((idx1 + idx2) / 2 - x))
                    min_between = df.iloc[mid_idx]['low']
                    
                    # Check if price broke below this level after the second top
                    breakout_idx = min(idx2 + 15, len(df) - 1)
                    breakout_confirmed = any(df.iloc[idx2+1:breakout_idx+1]['close'] < min_between)
                    
                    # Add to patterns if not too similar to existing ones
                    is_duplicate = False
                    for pattern in patterns['double_tops']:
                        existing_idx1, existing_val1 = pattern['first']
                        existing_idx2, existing_val2 = pattern['second']
                        
                        # Check if new pattern overlaps with existing one
                        if (abs(idx1 - existing_idx1) <= 3 or 
                            abs(idx2 - existing_idx2) <= 3):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        patterns['double_tops'].append({
                            'first': (idx1, max1_val),
                            'second': (idx2, max2_val),
                            'confirmed': breakout_confirmed,
                            'confirmation_price': min_between
                        })
    
    print(f"Found {len(patterns['double_bottoms'])} double bottoms and {len(patterns['double_tops'])} double tops.")
    return patterns

def plot_chart(df, patterns, symbol, timeframe):
    """
    Plot candlestick chart with double bottom and double top patterns
    """
    # Use mplfinance for better candlestick charts
    try:
        import mplfinance as mpf
        has_mplfinance = True
    except ImportError:
        has_mplfinance = False
        print("For better charts, install mplfinance: pip install mplfinance")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.title(f"{symbol} {timeframe} - Double Bottom/Top Patterns", fontsize=16)
    
    # Plot candlesticks
    if has_mplfinance:
        # Convert to mplfinance format
        df_mpf = df.copy()
        df_mpf = df_mpf.set_index('timestamp')
        df_mpf.index.name = 'Date'
        
        # Create custom style
        mc = mpf.make_marketcolors(up='green', down='red', 
                                   wick={'up':'green', 'down':'red'},
                                   edge={'up':'green', 'down':'red'},
                                   volume={'up':'green', 'down':'red'})
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=False)
        
        # Plot on existing axis
        mpf.plot(df_mpf, type='candle', style=s, ax=ax, volume=False)
    else:
        # Fallback to manual candlestick plotting
        width = 0.6
        width2 = width / 2
        
        up = df[df.close >= df.open]
        down = df[df.close < df.open]
        
        # Plot up candles
        ax.bar(up.index, up.close - up.open, width, bottom=up.open, color='green', alpha=0.7)
        ax.bar(up.index, up.high - up.close, width2, bottom=up.close, color='green', alpha=0.7)
        ax.bar(up.index, up.low - up.open, width2, bottom=up.open, color='green', alpha=0.7)
        
        # Plot down candles
        ax.bar(down.index, down.close - down.open, width, bottom=down.open, color='red', alpha=0.7)
        ax.bar(down.index, down.high - down.open, width2, bottom=down.open, color='red', alpha=0.7)
        ax.bar(down.index, down.low - down.close, width2, bottom=down.close, color='red', alpha=0.7)
    
    # Format x-axis with fewer labels to avoid overcrowding
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Show label every 6 hours
    plt.xticks(rotation=45)
    
    # Create pattern collections to avoid too many overlapping labels
    pattern_areas = []  # Keep track of areas with patterns to avoid overlapping labels
    pattern_bottoms = []
    pattern_tops = []
    
    # Draw double bottoms with minimal labels
    for i, pattern in enumerate(patterns['double_bottoms']):
        idx1, price1 = pattern['first']
        idx2, price2 = pattern['second']
        
        color = 'green' if pattern['confirmed'] else 'orange'
        
        # Draw line connecting the bottoms
        ax.plot([idx1, idx2], [price1, price2], color=color, linewidth=2)
        
        # Mark the bottoms
        ax.scatter([idx1, idx2], [price1, price2], color=color, s=100, zorder=5)
        
        # Add label for just a few patterns to avoid clutter
        if i < 3:  # Only label the first 3 patterns
            label_x = (idx1 + idx2) / 2
            label_y = min(price1, price2) * 0.995
            
            # Check if label position overlaps with existing labels
            overlapping = False
            for area in pattern_areas:
                x_range, y_range = area
                if (label_x >= x_range[0] and label_x <= x_range[1] and
                    label_y >= y_range[0] and label_y <= y_range[1]):
                    overlapping = True
                    break
            
            if not overlapping:
                ax.text(label_x, label_y, 'DB', 
                     horizontalalignment='center', color=color, 
                     fontweight='bold', fontsize=10)
                
                # Add to pattern areas to avoid overlap
                pattern_areas.append(((idx1-5, idx2+5), (label_y*0.99, label_y*1.01)))
                pattern_bottoms.append((idx1, idx2, price1, price2, pattern['confirmed']))
        
        # Draw neckline at confirmation price if confirmed
        if pattern['confirmed']:
            ax.axhline(y=pattern['confirmation_price'], 
                     xmin=max(0, (idx1-5)/len(df)), 
                     xmax=min(1, (idx2+15)/len(df)), 
                     color=color, linestyle='--', alpha=0.7)
    
    # Draw double tops with minimal labels
    for i, pattern in enumerate(patterns['double_tops']):
        idx1, price1 = pattern['first']
        idx2, price2 = pattern['second']
        
        color = 'red' if pattern['confirmed'] else 'orange'
        
        # Draw line connecting the tops
        ax.plot([idx1, idx2], [price1, price2], color=color, linewidth=2)
        
        # Mark the tops
        ax.scatter([idx1, idx2], [price1, price2], color=color, s=100, zorder=5)
        
        # Add label for just a few patterns to avoid clutter
        if i < 3:  # Only label the first 3 patterns
            label_x = (idx1 + idx2) / 2
            label_y = max(price1, price2) * 1.005
            
            # Check if label position overlaps with existing labels
            overlapping = False
            for area in pattern_areas:
                x_range, y_range = area
                if (label_x >= x_range[0] and label_x <= x_range[1] and
                    label_y >= y_range[0] and label_y <= y_range[1]):
                    overlapping = True
                    break
            
            if not overlapping:
                ax.text(label_x, label_y, 'DT', 
                     horizontalalignment='center', color=color, 
                     fontweight='bold', fontsize=10)
                
                # Add to pattern areas to avoid overlap
                pattern_areas.append(((idx1-5, idx2+5), (label_y*0.99, label_y*1.01)))
                pattern_tops.append((idx1, idx2, price1, price2, pattern['confirmed']))
        
        # Draw neckline at confirmation price if confirmed
        if pattern['confirmed']:
            ax.axhline(y=pattern['confirmation_price'], 
                     xmin=max(0, (idx1-5)/len(df)), 
                     xmax=min(1, (idx2+15)/len(df)), 
                     color=color, linestyle='--', alpha=0.7)
    
    # Add legend
    confirmed_bottom = patches.Patch(color='green', label='Confirmed Double Bottom (DB)')
    confirmed_top = patches.Patch(color='red', label='Confirmed Double Top (DT)')
    unconfirmed = patches.Patch(color='orange', label='Unconfirmed Pattern')
    ax.legend(handles=[confirmed_bottom, confirmed_top, unconfirmed], loc='upper right')
    
    # Print pattern summary in the console instead of cluttering the chart
    print("\nDouble Bottom Patterns:")
    for i, (idx1, idx2, price1, price2, confirmed) in enumerate(pattern_bottoms):
        status = "CONFIRMED" if confirmed else "unconfirmed"
        print(f"  {i+1}. Points at indices {idx1}, {idx2} with prices {price1:.2f}, {price2:.2f} - {status}")
    
    print("\nDouble Top Patterns:")
    for i, (idx1, idx2, price1, price2, confirmed) in enumerate(pattern_tops):
        status = "CONFIRMED" if confirmed else "unconfirmed"
        print(f"  {i+1}. Points at indices {idx1}, {idx2} with prices {price1:.2f}, {price2:.2f} - {status}")
    
    # Set y-axis to show more context
    buffer = (df['high'].max() - df['low'].min()) * 0.05
    ax.set_ylim(df['low'].min() - buffer, df['high'].max() + buffer)
    
    # Add grid but make it subtle
    ax.grid(alpha=0.2)
    plt.tight_layout()
    
    # Save the chart
    plt.savefig('double_pattern_chart.png', dpi=150)
    print("\nChart saved as double_pattern_chart.png")
    
    # Show the chart
    plt.show()

def main():
    # Set parameters
    symbol = 'BTC/USDT'
    timeframe = '5m'
    limit = 200
    
    # Optional parameters - adjust these to control sensitivity
    window_size = 8          # Max distance between pattern points (in candles)
    threshold = 1          # Price similarity threshold (1%)
    significance_pct = 0.5    # Minimum swing size to be considered significant (0.5%)
    
    # Fetch data
    df = fetch_ohlcv_data(symbol, timeframe, limit)
    
    if df is not None:
        # Find patterns with custom parameters
        patterns = find_patterns(df, window_size, threshold, significance_pct)
        
        # Plot chart
        plot_chart(df, patterns, symbol, timeframe)

if __name__ == "__main__":
    main()