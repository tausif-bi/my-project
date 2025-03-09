# simple_integration.py
import schedule
import time
import logging
from datetime import datetime

# Import functions directly
from trendline_breakout import fetch_and_check_breakouts
from rsi_trendline_breakout import create_dashboard
from rolling_window import scheduled_multi_task

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

def run_all_strategies():
    """Run all three trading strategies"""
    logger.info("Running all trading strategies")
    
    logger.info("Starting trendline breakout analysis")
    try:
        fetch_and_check_breakouts()
        logger.info("Trendline breakout completed")
    except Exception as e:
        logger.error(f"Error running trendline breakout: {e}")
    
    logger.info("Starting RSI trendline breakout analysis")
    try:
        create_dashboard()
        logger.info("RSI trendline breakout completed")
    except Exception as e:
        logger.error(f"Error running RSI trendline breakout: {e}")
    
    logger.info("Starting rolling window analysis")
    try:
        scheduled_multi_task()
        logger.info("Rolling window completed")
    except Exception as e:
        logger.error(f"Error running rolling window: {e}")
    
    logger.info("All trading strategies completed")

if __name__ == "__main__":
    # Run immediately on startup
    run_all_strategies()
    
    # Schedule to run every 15 minutes
    schedule.every(15).minutes.do(run_all_strategies)
    
    logger.info("Scheduler started. Press Ctrl+C to exit.")
    
    # Keep the script running
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")