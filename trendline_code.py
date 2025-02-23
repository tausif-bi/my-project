import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans

# Initialize Binance Exchange
exchange = ccxt.binance()

# Fetch 100 candles of BTC/USDT (1-hour timeframe)
symbol = 'BTC/USDT'
timeframe = '1h'
limit = 100  # Number of candles

# Fetch OHLCV data
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

# Convert to DataFrame
df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

# Convert timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

# Set index to Timestamp
df.set_index('Timestamp', inplace=True)

# Identify Local Highs (Maxima)
window = 5  # Adjust for smoother trend detection
df['Highs'] = df['Close'].iloc[argrelextrema(df['Close'].values, np.greater, order=window)]

# Drop NaNs to get valid high points
highs = df.dropna(subset=['Highs'])

# Convert datetime index to numerical values
x_highs = np.arange(len(df))[df['Highs'].notna()]
y_highs = highs['Highs'].values

# Compute slopes between consecutive highs
slopes = np.diff(y_highs) / np.diff(x_highs)

# Reshape slopes for clustering
slope_data = np.array(slopes).reshape(-1, 1)

# Use KMeans to classify slopes into clusters
n_clusters = 3  # Adjust the number of trend clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(slope_data)

# Assign each segment of the trendline to a cluster
trend_clusters = [[] for _ in range(n_clusters)]
for i in range(len(labels)):
    trend_clusters[labels[i]].append((x_highs[i], y_highs[i]))
    trend_clusters[labels[i]].append((x_highs[i + 1], y_highs[i + 1]))

# Extend Trendlines into the Future
future_steps = 10  # Number of future candles to extend

# Plot BTC Closing Price
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], color='blue', linestyle='-', linewidth=2, label='BTC/USDT Close Price')

# Plot Highs
plt.scatter(highs.index, highs['Highs'], color='green', marker='^', s=100, label='Highs')

# Assign unique colors to each trend cluster
colors = ['red', 'orange', 'purple', 'cyan', 'magenta']

# Plot multiple trendlines and extend them
for i, cluster in enumerate(trend_clusters):
    cluster = sorted(set(cluster))  # Remove duplicates
    x_cluster = [df.index[idx] for idx, _ in cluster]
    y_cluster = [y for _, y in cluster]

    if len(x_cluster) > 1:
        # Compute slope and intercept for extension
        slope = (y_cluster[-1] - y_cluster[0]) / (len(x_cluster) - 1)
        intercept = y_cluster[0] - slope * 0  # Trendline starts from first high

        # Extend trendline
        extended_x = list(x_cluster)
        extended_y = list(y_cluster)

        # Project future points
        last_index = df.index[-1]
        for step in range(1, future_steps + 1):
            next_time = last_index + pd.Timedelta(hours=step)
            next_price = y_cluster[-1] + (slope * step)
            extended_x.append(next_time)
            extended_y.append(next_price)

        plt.plot(extended_x, extended_y, color=colors[i % len(colors)], linestyle='--', linewidth=2, label=f'Extended Trendline {i+1}')

# Labels and Styling
plt.xlabel('Timestamp')
plt.ylabel('Price (USDT)')
plt.title('BTC/USDT 1H Closing Prices with Extended Trendlines')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
