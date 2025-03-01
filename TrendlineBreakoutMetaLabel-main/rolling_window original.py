import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    # Instead of reading from CSV, we use ccxt to fetch data from Binance.
    exchange = ccxt.binance({'enableRateLimit': True})
    symbol = 'BTC/USDT'
    timeframe = '5m'  # 1-hour candles
    limit = 500      # Fetch up to 1000 candles
    
    # Fetch OHLCV data: [timestamp, open, high, low, close, volume]
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    
    # Create DataFrame with appropriate column names
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data.set_index('date')
    
    # For hourly data, you may want to adjust the order parameter
    # Higher order = fewer but more significant extremes
    order = 4
    
    # Compute rolling window extremes on the close prices
    tops, bottoms = rw_extremes(data['close'].to_numpy(), order)
    
    # Print tops with their dates
    for top in tops:
        print("Top index:", top[1], "Date:", data.index[top[1]], "Price:", top[2])

    print("Data start:", data.index[0])
    print("Data end:", data.index[-1])
    
    # Set figure size for better visibility
    plt.figure(figsize=(14, 8))
    
    # Plot the close prices
    plt.plot(data.index, data['close'], linewidth=1)
    
    # Enhanced marker size and visibility
    marker_size = 100  # Increase marker size
    alpha_value = 0.7  # Semi-transparent markers
    
    # Plot tops with enhanced markers
    for top in tops:
        plt.scatter(data.index[top[1]], top[2], 
                   s=marker_size, marker='^', color='green', 
                   alpha=alpha_value, zorder=5)
    
    # Plot bottoms with enhanced markers
    for bottom in bottoms:
        plt.scatter(data.index[bottom[1]], bottom[2], 
                   s=marker_size, marker='v', color='red', 
                   alpha=alpha_value, zorder=5)
    
    # Improve plot formatting
    plt.title(f'BTC/USDT {timeframe} Price with Local Extremes (Order={order})')
    plt.ylabel('Price (USDT)')
    plt.grid(True, alpha=0.3)
    
    # Ensure dates are formatted properly on x-axis
    plt.gcf().autofmt_xdate()
    
    plt.show()