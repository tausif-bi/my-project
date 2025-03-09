# integrated_signals.py
import schedule
import time
import logging
import os
import subprocess
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_signals.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("trading_signals")

def run_trendline_breakout():
    """Run the trendline breakout script and log results"""
    logger.info("Starting trendline breakout analysis")
    try:
        result = subprocess.run(
            ["python", "trendline_breakout.py"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info(f"Trendline breakout completed: {result.stdout}")
        else:
            logger.error(f"Trendline breakout failed with error: {result.stderr}")
    except Exception as e:
        logger.error(f"Error running trendline breakout: {e}")

def run_rsi_trendline_breakout():
    """Run the RSI trendline breakout script and log results"""
    logger.info("Starting RSI trendline breakout analysis")
    try:
        result = subprocess.run(
            ["python", "rsi_trendline_breakout.py"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info(f"RSI trendline breakout completed: {result.stdout}")
        else:
            logger.error(f"RSI trendline breakout failed with error: {result.stderr}")
    except Exception as e:
        logger.error(f"Error running RSI trendline breakout: {e}")

def run_rolling_window():
    """Run the rolling window script and log results"""
    logger.info("Starting rolling window analysis")
    try:
        result = subprocess.run(
            ["python", "rolling_window.py"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info(f"Rolling window completed: {result.stdout}")
        else:
            logger.error(f"Rolling window failed with error: {result.stderr}")
    except Exception as e:
        logger.error(f"Error running rolling window: {e}")

def run_all_strategies():
    """Run all three trading strategies"""
    logger.info("Running all trading strategies")
    run_trendline_breakout()
    run_rsi_trendline_breakout()
    run_rolling_window()
    logger.info("All trading strategies completed")

if __name__ == "__main__":
    # Run immediately on startup
    run_all_strategies()
    
    # Schedule to run every hour
    schedule.every(15).minutes.do(run_all_strategies)
    
    logger.info("Scheduler started. Press Ctrl+C to exit.")
    
    # Keep the script running
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")