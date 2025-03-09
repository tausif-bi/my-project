# db_utils.py
import mysql.connector
import os
from datetime import datetime
import json

# Database configuration
DB_CONFIG = {
    'host': '164.68.111.47',
    'port': '3306',
    'user': 'tausif',
    'password': 'A852741z',
    'database': 'trading_signals'
}

def get_db_connection():
    """Create and return a MySQL database connection"""
    conn = mysql.connector.connect(**DB_CONFIG)
    return conn

def store_signal(symbol, timeframe, signal_time, price, 
                signal_type, signal_details, direction=None, 
                level_value=None, level_type=None, cross_type=None,
                rsi_value=None, chart_image_path=None):
    """
    Store a trading signal in the database
    
    Parameters:
    - symbol: Trading pair (e.g., 'BTC/USDT')
    - timeframe: Timeframe (e.g., '1h', '5m')
    - signal_time: Timestamp of the signal
    - price: Price at the time of the signal
    - signal_type: One of 'trendline_breakout', 'rsi_breakout', or 'swing_high_low'
    - signal_details: Text description of the signal
    - direction: 'upward' or 'downward'
    - level_value: Value of support/resistance level
    - level_type: Type of level (e.g., 'support', 'resistance')
    - cross_type: Type of cross (e.g., 'Support Breakout', 'Resistance Rejection')
    - rsi_value: RSI value (for RSI signals)
    - chart_image_path: Path to saved chart image
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Clean symbol (remove '/' for consistency)
        clean_symbol = symbol.replace('/', '')
        
        # Check if a signal already exists for this symbol, timeframe, and timestamp
        cursor.execute("""
            SELECT id FROM signals 
            WHERE symbol = %s AND timeframe = %s AND signal_time = %s
        """, (clean_symbol, timeframe, signal_time))
        
        result = cursor.fetchone()
        
        if result:
            # Signal exists - update it
            signal_id = result[0]
            
            # Update specific fields based on signal type
            if signal_type == 'trendline_breakout':
                cursor.execute("""
                    UPDATE signals 
                    SET trendline_breakout = TRUE, 
                        trendline_breakout_details = %s,
                        signal_direction = %s,
                        level_value = %s,
                        level_type = %s,
                        cross_type = %s,
                        chart_image_path = COALESCE(%s, chart_image_path)
                    WHERE id = %s
                """, (signal_details, direction, level_value, level_type, cross_type, chart_image_path, signal_id))
            
            elif signal_type == 'rsi_breakout':
                cursor.execute("""
                    UPDATE signals 
                    SET rsi_breakout = TRUE, 
                        rsi_breakout_details = %s,
                        rsi_value = %s,
                        signal_direction = %s,
                        level_value = %s,
                        level_type = %s,
                        cross_type = %s,
                        chart_image_path = COALESCE(%s, chart_image_path)
                    WHERE id = %s
                """, (signal_details, rsi_value, direction, level_value, level_type, cross_type, chart_image_path, signal_id))
            
            elif signal_type == 'swing_high_low':
                cursor.execute("""
                    UPDATE signals 
                    SET swing_high_low = TRUE, 
                        swing_high_low_details = %s,
                        signal_direction = %s,
                        level_value = %s,
                        level_type = %s,
                        cross_type = %s,
                        chart_image_path = COALESCE(%s, chart_image_path)
                    WHERE id = %s
                """, (signal_details, direction, level_value, level_type, cross_type, chart_image_path, signal_id))
        
        else:
            # Create new signal
            # Set all signal-specific columns to default values first
            query = """
                INSERT INTO signals (
                    symbol, timeframe, signal_time, price,
                    trendline_breakout, trendline_breakout_details,
                    rsi_breakout, rsi_breakout_details,
                    swing_high_low, swing_high_low_details,
                    rsi_value, signal_direction, level_value, level_type, cross_type,
                    chart_image_path
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Initialize all values to defaults
            values = [
                clean_symbol, timeframe, signal_time, price,
                False, None,  # trendline_breakout fields
                False, None,  # rsi_breakout fields
                False, None,  # swing_high_low fields
                rsi_value, direction, level_value, level_type, cross_type,
                chart_image_path
            ]
            
            # Set the specific signal type to True and its details
            if signal_type == 'trendline_breakout':
                values[4] = True  # trendline_breakout = True
                values[5] = signal_details  # trendline_breakout_details
            elif signal_type == 'rsi_breakout':
                values[6] = True  # rsi_breakout = True
                values[7] = signal_details  # rsi_breakout_details
            elif signal_type == 'swing_high_low':
                values[8] = True  # swing_high_low = True
                values[9] = signal_details  # swing_high_low_details
            
            cursor.execute(query, values)
            signal_id = cursor.lastrowid
        
        conn.commit()
        print(f"Signal stored successfully: {signal_type} for {symbol} at {signal_time}")
        return signal_id
    
    except Exception as e:
        conn.rollback()
        print(f"Error storing signal: {e}")
        return None
    
    finally:
        cursor.close()
        conn.close()

def parse_signal_timestamp(timestamp_str):
    """Parse timestamp from signal string"""
    try:
        # Handle different formats
        formats = [
            '%Y-%m-%d %H:%M:%S',  # 2025-03-01 11:11:00
            '%Y-%m-%d %H:%M'      # 2025-03-01 11:11
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Could not parse timestamp: {timestamp_str}")
        
    except Exception as e:
        print(f"Error parsing timestamp: {e}")
        return None

def save_chart_image(symbol, timeframe, chart_data, directory="chart_images"):
    """Save chart image and return the path"""
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Create subdirectory by date
    date_dir = os.path.join(directory, datetime.now().strftime('%Y%m%d'))
    os.makedirs(date_dir, exist_ok=True)
    
    # Clean symbol for filename
    clean_symbol = symbol.replace('/', '_')
    
    # Generate filename
    filename = f"{clean_symbol}_{timeframe}_{datetime.now().strftime('%H%M%S')}.png"
    filepath = os.path.join(date_dir, filename)
    
    # Save the image
    with open(filepath, 'wb') as f:
        f.write(chart_data)
    
    # Return relative path for database
    return os.path.join(datetime.now().strftime('%Y%m%d'), filename)