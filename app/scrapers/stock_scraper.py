"""
Stock data scraper for comprehensive stock market data collection
"""
import asyncio
from typing import List, Dict, Any
from app.services.stock_service import get_nifty_500_stocks
from app.scrapers.stock_scraper_db import insert_stock_data, get_stock_data_from_db

async def scrape_all_stocks() -> Dict[str, Any]:
    """
    Scrape all stock data asynchronously
    Returns a dictionary with status and results
    """
    try:
        print("Starting comprehensive stock data scraping...")
        
        # Get stock data from multiple sources
        stocks = await asyncio.to_thread(get_nifty_500_stocks)
        
        if isinstance(stocks, list) and len(stocks) > 0:
            # Store in database
            success = await asyncio.to_thread(insert_stock_data, stocks)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Successfully scraped and stored {len(stocks)} stock records",
                    "count": len(stocks),
                    "timestamp": None
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to store stock data in database",
                    "count": 0
                }
        else:
            return {
                "status": "error",
                "message": "No stock data retrieved from sources",
                "count": 0
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Stock scraping failed: {str(e)}",
            "count": 0
        }

def scrape_stocks_sync() -> Dict[str, Any]:
    """
    Synchronous version for backward compatibility
    """
    try:
        stocks = get_nifty_500_stocks()
        if isinstance(stocks, list) and len(stocks) > 0:
            success = insert_stock_data(stocks)
            if success:
                return {
                    "status": "success",
                    "message": f"Successfully scraped and stored {len(stocks)} stock records",
                    "count": len(stocks)
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to store stock data",
                    "count": 0
                }
        else:
            return {
                "status": "error",
                "message": "No stock data retrieved",
                "count": 0
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Stock scraping failed: {str(e)}",
            "count": 0
        }
