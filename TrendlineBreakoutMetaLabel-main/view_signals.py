# view_signals.py
from db_utils import get_db_connection
from datetime import datetime, timedelta
import argparse

def print_recent_signals(hours=24, symbol=None, timeframe=None, signal_type=None):
    """Print recent signals from the database"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        query = """
        SELECT 
            symbol, timeframe, signal_time, price,
            trendline_breakout, trendline_breakout_details,
            rsi_breakout, rsi_breakout_details,
            swing_high_low, swing_high_low_details,
            rsi_value, signal_direction, level_value, level_type, cross_type
        FROM signals
        WHERE signal_time >= DATE_SUB(NOW(), INTERVAL %s HOUR)
        """
        
        params = [hours]
        
        if symbol:
            query += " AND symbol = %s"
            params.append(symbol.replace('/', ''))
        
        if timeframe:
            query += " AND timeframe = %s"
            params.append(timeframe)
        
        if signal_type:
            if signal_type == 'trendline':
                query += " AND trendline_breakout = TRUE"
            elif signal_type == 'rsi':
                query += " AND rsi_breakout = TRUE"
            elif signal_type == 'swing':
                query += " AND swing_high_low = TRUE"
        
        query += " ORDER BY signal_time DESC"
        
        cursor.execute(query, params)
        signals = cursor.fetchall()
        
        if not signals:
            print(f"No signals found in the last {hours} hours")
            return
        
        print(f"Found {len(signals)} signals in the last {hours} hours:")
        print("-" * 100)
        
        # Print signals in tabular format
        print(f"{'Symbol':<10} {'Timeframe':<8} {'Signal Time':<20} {'Signal Type':<15} {'Direction':<10} {'Price':<12}")
        print("-" * 100)
        
        for signal in signals:
            signal_types = []
            
            if signal['trendline_breakout']:
                signal_types.append("Trendline")
            if signal['rsi_breakout']:
                signal_types.append("RSI")
            if signal['swing_high_low']:
                signal_types.append("Swing")
                
            signal_type_str = ", ".join(signal_types)
            
            print(f"{signal['symbol']:<10} {signal['timeframe']:<8} {signal['signal_time']!s:<20} {signal_type_str:<15} {signal['signal_direction'] or 'N/A':<10} {signal['price']!s:<12}")
            
            # Print details
            if signal['trendline_breakout'] and signal['trendline_breakout_details']:
                print(f"  Trendline: {signal['trendline_breakout_details']}")
            if signal['rsi_breakout'] and signal['rsi_breakout_details']:
                print(f"  RSI: {signal['rsi_breakout_details']}")
            if signal['swing_high_low'] and signal['swing_high_low_details']:
                print(f"  Swing: {signal['swing_high_low_details']}")
                
            print("-" * 100)
            
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View recent trading signals")
    parser.add_argument("--hours", type=int, default=24, help="Hours to look back")
    parser.add_argument("--symbol", type=str, help="Filter by symbol")
    parser.add_argument("--timeframe", type=str, help="Filter by timeframe")
    parser.add_argument("--type", type=str, choices=['trendline', 'rsi', 'swing'], help="Filter by signal type")
    
    args = parser.parse_args()
    
    print_recent_signals(args.hours, args.symbol, args.timeframe, args.type)