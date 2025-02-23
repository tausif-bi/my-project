import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Download Apple stock price data
df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# Calculate daily returns
df['Returns'] = df['Close'].pct_change()

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Returns'], label="Stock Returns", color="blue")
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('Apple (AAPL) Daily Returns - Stationary Time Series')
plt.legend()
plt.show()
