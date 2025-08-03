from fastapi import APIRouter, HTTPException
from app.services.stock_service import get_nifty_500_stocks, get_all_nse_stocks
import yfinance as yf

router = APIRouter(prefix="/stocks", tags=["Stocks"])

@router.get("/live")
def get_nifty_stocks():
    """Get NIFTY 500 stocks (with fallback if NSE blocks)"""
    result = get_nifty_500_stocks()
    return {
        "source": "Yahoo Finance real-time data",
        "count": len(result) if isinstance(result, list) else 0,
        "data": result
    }

@router.get("/yfinance")
def get_stocks_yfinance():
    """Get stock data using YFinance (more reliable alternative)"""
    try:
        # Popular NSE stocks
        symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS']
        stocks = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    
                    stocks.append({
                        "symbol": symbol.replace('.NS', ''),
                        "companyName": info.get('longName', symbol),
                        "lastPrice": round(float(current_price), 2),
                        "marketCap": info.get('marketCap', 0),
                        "sector": info.get('sector', 'Unknown'),
                        "source": "Yahoo Finance"
                    })
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue
        
        return {
            "source": "Yahoo Finance API",
            "count": len(stocks),
            "data": stocks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YFinance error: {str(e)}")

@router.get("/all")
def get_all_stocks():
    """Get all NSE stocks (slower, from CSV file)"""
    try:
        result = get_all_nse_stocks()
        if isinstance(result, dict) and result.get("error"):
            raise HTTPException(status_code=503, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
def get_stocks_status():
    """Quick status check for stocks endpoints"""
    return {
        "service": "stocks",
        "status": "active",
        "available_endpoints": {
            "/stocks": "Real Yahoo Finance stock data",
            "/stocks/live": "Live stock data from database",
            "/stocks/all": "All available stock data"
        }
    }