import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks
import ccxt
from datetime import datetime
import mplfinance as mpf

def get_price_data(symbol='BTC/USDT', timeframe='1d', limit=365):
    """Fetch price data from exchange."""
    exchange = ccxt.binance({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def find_significant_points(prices, prominence=0.03, distance=10):
    """Find significant highs and lows in the price series."""
    # Normalize prices for better peak detection
    norm_prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))
    
    # Find peaks (highs)
    highs_idx, _ = find_peaks(norm_prices, prominence=prominence, distance=distance)
    
    # Find troughs (lows) by inverting the signal
    lows_idx, _ = find_peaks(-norm_prices, prominence=prominence, distance=distance)
    
    return highs_idx, lows_idx

def identify_elliott_waves(prices, highs_idx, lows_idx):
    """Identify potential Elliott Wave patterns."""
    # Create a list of all significant points
    all_points = []
    for idx in highs_idx:
        if idx < len(prices):
            all_points.append((idx, prices[idx], 'high'))
    
    for idx in lows_idx:
        if idx < len(prices):
            all_points.append((idx, prices[idx], 'low'))
    
    all_points.sort(key=lambda x: x[0])
    
    print(f"Found {len(all_points)} significant points: {len(highs_idx)} highs and {len(lows_idx)} lows")
    
    waves = {
        "impulse": [],
        "corrective": [],
        "all_points": all_points
    }
    
    # Search for impulse patterns (5-wave structure)
    for i in range(len(all_points) - 5):
        points = all_points[i:i+6]  # Need 6 points for a complete impulse wave (0-1-2-3-4-5)
        
        # Check if the pattern alternates correctly (low-high-low-high-low-high)
        types = [p[2] for p in points]
        if types != ['low', 'high', 'low', 'high', 'low', 'high']:
            continue
        
        # Get prices and calculate price movements
        prices_segment = [p[1] for p in points]
        moves = [prices_segment[i+1] - prices_segment[i] for i in range(5)]
        
        # Apply Elliott Wave rules
        # 1. Waves 1, 3, 5 must be upward
        if moves[0] <= 0 or moves[2] <= 0 or moves[4] <= 0:
            continue
        
        # 2. Waves 2, 4 must be downward
        if moves[1] >= 0 or moves[3] >= 0:
            continue
        
        # 3. Wave 3 cannot be the shortest of waves 1, 3, 5
        if abs(moves[2]) < abs(moves[0]) and abs(moves[2]) < abs(moves[4]):
            continue
        
        # 4. Wave 2 cannot retrace beyond start of wave 1
        if prices_segment[2] <= prices_segment[0]:
            continue
        
        # 5. Wave 4 should not overlap with wave 1
        if prices_segment[4] <= prices_segment[1]:
            continue
        
        # We've found a potential impulse pattern
        wave_points = [
            ('0', points[0][0], points[0][1]),
            ('1', points[1][0], points[1][1]),
            ('2', points[2][0], points[2][1]),
            ('3', points[3][0], points[3][1]),
            ('4', points[4][0], points[4][1]),
            ('5', points[5][0], points[5][1])
        ]
        waves["impulse"].append(wave_points)
    
    # Search for corrective patterns (A-B-C)
    for i in range(len(all_points) - 3):
        points = all_points[i:i+4]  # Need 4 points for a complete corrective wave
        
        # Check if the pattern alternates correctly (high-low-high-low)
        types = [p[2] for p in points]
        if types != ['high', 'low', 'high', 'low']:
            continue
        
        # Get prices and calculate price movements
        prices_segment = [p[1] for p in points]
        moves = [prices_segment[i+1] - prices_segment[i] for i in range(3)]
        
        # Apply Elliott Wave rules
        # 1. Waves A and C must be downward
        if moves[0] >= 0 or moves[2] >= 0:
            continue
        
        # 2. Wave B must be upward
        if moves[1] <= 0:
            continue
        
        # 3. Wave B should generally not exceed the start of wave A
        if prices_segment[2] >= prices_segment[0]:
            continue
        
        # We've found a potential corrective pattern
        wave_points = [
            ('5/A', points[0][0], points[0][1]),
            ('A', points[1][0], points[1][1]),
            ('B', points[2][0], points[2][1]),
            ('C', points[3][0], points[3][1])
        ]
        waves["corrective"].append(wave_points)
    
    return waves

def plot_elliott_waves(df, waves, save_path=None):
    """Plot candlestick chart with Elliott Wave annotations."""
    # Set up the plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Plot candlestick chart (basic version without mplfinance for simplicity)
    # This is a simplified version - for production use mplfinance
    ax.plot(df.index, df['close'], color='white', linewidth=1)
    
    # Plot all significant points
    high_points = [(p[0], p[1]) for p in waves['all_points'] if p[2] == 'high']
    low_points = [(p[0], p[1]) for p in waves['all_points'] if p[2] == 'low']
    
    # Plot high points
    for idx, price in high_points:
        if idx < len(df):
            ax.scatter(df.index[idx], price, color='cyan', marker='^', s=100, alpha=0.7)
    
    # Plot low points
    for idx, price in low_points:
        if idx < len(df):
            ax.scatter(df.index[idx], price, color='magenta', marker='v', s=100, alpha=0.7)
    
    # Plot impulse waves
    for wave in waves.get("impulse", []):
        indices = [point[1] for point in wave]
        values = [point[2] for point in wave]
        labels = [point[0] for point in wave]
        
        # Get valid dates
        valid_dates = []
        valid_values = []
        valid_labels = []
        
        for i, idx in enumerate(indices):
            if idx < len(df):
                valid_dates.append(df.index[idx])
                valid_values.append(values[i])
                valid_labels.append(labels[i])
        
        # Plot lines and labels
        if len(valid_dates) > 1:
            ax.plot(valid_dates, valid_values, 'g-', linewidth=2, zorder=3)
            
            # Add labels
            for i, (date, value, label) in enumerate(zip(valid_dates, valid_values, valid_labels)):
                ax.annotate(label, (date, value), xytext=(0, 5 if i % 2 else -15), 
                           textcoords='offset points', fontsize=12, fontweight='bold',
                           color='white', bbox=dict(facecolor='green', alpha=0.7))
    
    # Plot corrective waves
    for wave in waves.get("corrective", []):
        indices = [point[1] for point in wave]
        values = [point[2] for point in wave]
        labels = [point[0] for point in wave]
        
        # Get valid dates
        valid_dates = []
        valid_values = []
        valid_labels = []
        
        for i, idx in enumerate(indices):
            if idx < len(df):
                valid_dates.append(df.index[idx])
                valid_values.append(values[i])
                valid_labels.append(labels[i])
        
        # Plot lines and labels
        if len(valid_dates) > 1:
            ax.plot(valid_dates, valid_values, 'r-', linewidth=2, zorder=3)
            
            # Add labels
            for i, (date, value, label) in enumerate(zip(valid_dates, valid_values, valid_labels)):
                ax.annotate(label, (date, value), xytext=(0, 5 if i % 2 == 0 else -15), 
                           textcoords='offset points', fontsize=12, fontweight='bold',
                           color='white', bbox=dict(facecolor='red', alpha=0.7))
    
    # Set title and labels
    ax.set_title('Elliott Wave Analysis', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    
    # Format date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='g', lw=2, label='Impulse Waves (1-2-3-4-5)'),
        Line2D([0], [0], color='r', lw=2, label='Corrective Waves (A-B-C)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='cyan', markersize=8, label='Significant Highs'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='magenta', markersize=8, label='Significant Lows')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Tighten layout
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Chart saved to {save_path}")
    
    return fig, ax

def analyze_elliott_waves(symbol='BTC/USDT', timeframe='1d', limit=365, 
                        prominence=0.03, distance=10, save_path=None):
    """
    Analyze recent price data for Elliott Wave patterns.
    
    Parameters:
    - symbol: Trading pair
    - timeframe: Candle timeframe ('1d', '4h', etc.)
    - limit: Number of candles to fetch
    - prominence: Peak prominence parameter
    - distance: Minimum distance between peaks
    - save_path: Path to save the chart
    
    Returns:
    - DataFrame with price data
    - Dictionary of identified wave patterns
    - Tuple of (figure, axes)
    """
    print(f"Analyzing Elliott Waves for {symbol} ({timeframe})...")
    print(f"Fetching {limit} candles...")
    
    # Get price data
    df = get_price_data(symbol, timeframe, limit)
    print(f"Data loaded: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Find significant points
    print(f"Finding significant points (prominence={prominence}, distance={distance})...")
    highs_idx, lows_idx = find_significant_points(df['close'].values, prominence, distance)
    
    # Identify Elliott Wave patterns
    print("Analyzing Elliott Wave patterns...")
    waves = identify_elliott_waves(df['close'].values, highs_idx, lows_idx)
    
    # Count patterns found
    impulse_count = len(waves.get("impulse", []))
    corrective_count = len(waves.get("corrective", []))
    print(f"Found {impulse_count} potential impulse patterns and {corrective_count} potential corrective patterns")
    
    # Plot the results
    print("Generating chart with Elliott Wave annotations...")
    fig, ax = plot_elliott_waves(df, waves, save_path)
    
    return df, waves, (fig, ax)

if __name__ == "__main__":
    # Configure the analysis
    symbol = 'BTC/USDT'
    timeframe = '1d'
    limit = 365  # Last year of data
    
    # Parameters for Elliott Wave detection
    prominence = 0.03  # How prominent a peak should be
    distance = 10      # Minimum number of candles between peaks
    
    # Run the analysis
    df, waves, (fig, ax) = analyze_elliott_waves(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
        prominence=prominence,
        distance=distance,
        save_path=f"{symbol.replace('/', '_')}_{timeframe}_elliott_waves.png"
    )
    
    plt.show()