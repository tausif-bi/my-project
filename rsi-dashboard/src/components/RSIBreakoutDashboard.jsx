import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Scatter } from 'recharts';

const RSIBreakoutDashboard = () => {
  const [coins, setCoins] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Mock data generation to simulate the RSI breakout chart
  useEffect(() => {
    const generateCoinData = () => {
      const symbols = [
        'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'BNB/USDT', 
        'SOL/USDT', 'ADA/USDT', 'TRX/USDT', 'LINK/USDT'
      ];
      
      const coinData = symbols.map(symbol => {
        // Generate 30 data points for each coin (simulating the last 30 candles)
        const dataPoints = Array.from({ length: 30 }, (_, i) => {
          // Start with a base RSI and add some randomness
          const baseRSI = 40 + Math.random() * 20;
          let rsi = baseRSI + (Math.sin(i / 5) * 10) + (Math.random() * 5 - 2.5);
          
          // Ensure RSI stays within 0-100 range
          rsi = Math.max(0, Math.min(100, rsi));
          
          // Generate support and resistance lines with slight incline/decline
          const supportSlope = (Math.random() * 0.3) - 0.15;
          const resistSlope = (Math.random() * 0.3) - 0.15;
          const supportBase = baseRSI - 10;
          const resistBase = baseRSI + 10;
          
          const support = supportBase + (i * supportSlope);
          const resistance = resistBase + (i * resistSlope);
          
          return {
            time: `${(i % 24).toString().padStart(2, '0')}:${(i * 2 % 60).toString().padStart(2, '0')}`,
            rsi: rsi,
            support: support,
            resistance: resistance,
            date: new Date(Date.now() - (30 - i) * 15 * 60000).toISOString()
          };
        });
        
        // Add breakout signals
        // Support breakout
        const supportBreakoutIndex = Math.floor(20 + Math.random() * 8);
        if (dataPoints[supportBreakoutIndex]) {
          dataPoints[supportBreakoutIndex].supportBreakout = dataPoints[supportBreakoutIndex].rsi;
        }
        
        // Resistance breakout (for some coins)
        if (Math.random() > 0.7) {
          const resistBreakoutIndex = Math.floor(22 + Math.random() * 6);
          if (dataPoints[resistBreakoutIndex]) {
            dataPoints[resistBreakoutIndex].resistBreakout = dataPoints[resistBreakoutIndex].rsi;
          }
        }
        
        // Current price
        const currentPrice = (symbol === 'BTC/USDT') ? 65000 + Math.random() * 1000 :
                           (symbol === 'ETH/USDT') ? 3500 + Math.random() * 100 :
                           (symbol === 'XRP/USDT') ? 0.55 + Math.random() * 0.05 :
                           (symbol === 'BNB/USDT') ? 580 + Math.random() * 20 :
                           (symbol === 'SOL/USDT') ? 140 + Math.random() * 10 :
                           (symbol === 'ADA/USDT') ? 0.45 + Math.random() * 0.05 :
                           (symbol === 'TRX/USDT') ? 0.12 + Math.random() * 0.01 :
                           (symbol === 'LINK/USDT') ? 16.5 + Math.random() * 0.5 : 10;
        
        // Determine if there's a recent signal
        const hasRecentSupportBreakout = dataPoints.slice(-3).some(d => d.supportBreakout);
        const hasRecentResistBreakout = dataPoints.slice(-3).some(d => d.resistBreakout);
        
        return {
          symbol,
          data: dataPoints,
          currentPrice: currentPrice.toFixed(symbol.includes('BTC') ? 0 : symbol.includes('ETH') || symbol.includes('BNB') || symbol.includes('SOL') ? 1 : 4),
          currentRSI: dataPoints[dataPoints.length - 1].rsi.toFixed(1),
          hasSignal: hasRecentSupportBreakout || hasRecentResistBreakout,
          signalType: hasRecentSupportBreakout ? 'support' : hasRecentResistBreakout ? 'resistance' : null
        };
      });
      
      return coinData;
    };
    
    try {
      setLoading(true);
      const data = generateCoinData();
      setCoins(data);
      setLoading(false);
    } catch (err) {
      setError("Failed to generate dashboard data");
      setLoading(false);
    }
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center h-64">Loading RSI Breakout Dashboard...</div>;
  }

  if (error) {
    return <div className="text-red-500 p-4">{error}</div>;
  }

  return (
    <div className="bg-gray-900 text-white p-4 rounded-lg">
      <h1 className="text-xl font-bold mb-4 text-center">RSI-10 Trendline Breakout Dashboard</h1>
      <div className="text-sm mb-4 text-center">
        {new Date().toLocaleString()} | 
        Active Signals: {coins.filter(c => c.hasSignal).length > 0 ? 
          coins.filter(c => c.hasSignal).map(c => 
            `${c.symbol}: ${c.signalType === 'support' ? 'Support' : 'Resistance'} Breakout - RSI:${c.currentRSI}`
          ).join(' | ') 
          : 'No active signals'}
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {coins.map(coin => (
          <div 
            key={coin.symbol} 
            className={`p-3 rounded-lg border ${
              coin.hasSignal && coin.signalType === 'support' ? 'border-green-500' :
              coin.hasSignal && coin.signalType === 'resistance' ? 'border-yellow-500' :
              'border-gray-700'
            }`}
          >
            <h2 className={`font-bold mb-2 ${
              coin.hasSignal && coin.signalType === 'support' ? 'text-green-400' :
              coin.hasSignal && coin.signalType === 'resistance' ? 'text-yellow-400' :
              'text-white'
            }`}>
              {coin.symbol} - RSI:{coin.currentRSI} - ${coin.currentPrice}
            </h2>
            
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={coin.data} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis 
                    dataKey="time" 
                    tick={{ fill: '#999' }} 
                    tickCount={5} 
                  />
                  <YAxis domain={[0, 100]} tick={{ fill: '#999' }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#333', border: 'none' }}
                    labelStyle={{ color: '#ccc' }}
                  />
                  <ReferenceLine y={70} stroke="red" strokeDasharray="3 3" />
                  <ReferenceLine y={30} stroke="green" strokeDasharray="3 3" />
                  <ReferenceLine y={50} stroke="yellow" strokeDasharray="3 3" />
                  <Line 
                    type="monotone" 
                    dataKey="support" 
                    stroke="red" 
                    strokeDasharray="3 3" 
                    strokeWidth={1} 
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="resistance" 
                    stroke="green" 
                    strokeDasharray="3 3" 
                    strokeWidth={1} 
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="rsi" 
                    stroke="magenta" 
                    strokeWidth={2} 
                    dot={false} 
                  />
                  <Scatter 
                    dataKey="supportBreakout" 
                    fill="lime" 
                    shape="triangle" 
                  />
                  <Scatter 
                    dataKey="resistBreakout" 
                    fill="yellow" 
                    shape="triangle" 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RSIBreakoutDashboard;