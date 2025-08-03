import yfinance as yf
from typing import List, Dict, Any
import time

def get_nifty_500_stocks():
    """Get optimized stock data from Yahoo Finance - focused on top Indian stocks"""
    try:
        return get_top_indian_stocks()
    except Exception as e:
        print(f"Yahoo Finance method failed: {e}")
        return {
            "status": "error",
            "message": f"Failed to fetch stock data from Yahoo Finance: {str(e)}",
            "error": "Real-time data unavailable",
            "count": 0,
            "stocks": []
        }

def get_top_indian_stocks():
    """Get top 100 Indian stocks from Yahoo Finance with optimized performance"""
    
    # Curated list of top 100 Indian stocks across sectors
    top_stocks = [
        # Top 20 by market cap
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "NESTLEIND.NS",
        "ULTRACEMCO.NS", "TITAN.NS", "WIPRO.NS", "BAJFINANCE.NS", "SUNPHARMA.NS",
        
        # Banking & Financial Services
        "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS", "INDUSINDBK.NS", "ICICIGI.NS",
        "HDFCAMC.NS", "PFC.NS", "RECLTD.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS",
        
        # Information Technology
        "HCLTECH.NS", "TECHM.NS", "LTI.NS", "MINDTREE.NS", "MPHASIS.NS",
        "PERSISTENT.NS", "LTTS.NS", "OFSS.NS", "COFORGE.NS", "NIITTECH.NS",
        
        # Energy & Utilities
        "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS", "IOC.NS",
        "BPCL.NS", "HINDPETRO.NS", "GAIL.NS", "TATAPOWER.NS", "ADANIPORTS.NS",
        
        # Consumer Goods & FMCG
        "BRITANNIA.NS", "DABUR.NS", "MARICO.NS", "GODREJCP.NS", "COLPAL.NS",
        "TATACONSUM.NS", "UBL.NS", "VBL.NS", "EMAMILTD.NS", "JYOTHYLAB.NS",
        
        # Automotive
        "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
        "ASHOKLEY.NS", "TVSMOTOR.NS", "ESCORTS.NS", "MRF.NS", "APOLLOTYRE.NS",
        
        # Metals & Mining
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "SAIL.NS",
        "JINDALSTEL.NS", "NMDC.NS", "MOIL.NS", "HINDZINC.NS", "NATIONALUM.NS",
        
        # Pharmaceuticals
        "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BIOCON.NS", 
        "TORNTPHARM.NS", "AUROPHARMA.NS", "LUPIN.NS", "ALKEM.NS", "GLENMARK.NS",
        
        # Cement & Construction
        "SHREECEM.NS", "GRASIM.NS", "ACC.NS", "AMBUJACEM.NS",
        
        # Telecom & Media
        "IDEA.NS", "ZEEL.NS", "SUNTV.NS", "NETWORK18.NS", "TVTODAY.NS"
    ]
    
    # Remove delisted stocks that cause errors
    delisted_stocks = ["LTI.NS", "MINDTREE.NS", "NIITTECH.NS", "CADILAHC.NS", "DALMIACEM.NS"]
    top_stocks = [stock for stock in top_stocks if stock not in delisted_stocks]
    
    print(f"Fetching data for {len(top_stocks)} top Indian stocks...")
    start_time = time.time()
    
    stocks = []
    
    # Process sequentially with delays to avoid rate limiting
    def fetch_single_stock(symbol, retry_count=0):
        try:
            ticker = yf.Ticker(symbol)
            # Get basic info with timeout
            info = ticker.get_info()
            hist = ticker.history(period="2d")
            
            if hist.empty:
                return None
                
            current_price = float(hist['Close'].iloc[-1])
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            change = current_price - prev_close
            pchange = (change / prev_close) * 100 if prev_close > 0 else 0
            
            return {
                "symbol": symbol.replace(".NS", ""),
                "companyName": info.get("longName", info.get("shortName", symbol.replace(".NS", ""))),
                "industry": info.get("industry", info.get("sector", "Unknown")),
                "lastPrice": round(current_price, 2),
                "dayHigh": round(float(hist['High'].iloc[-1]), 2),
                "dayLow": round(float(hist['Low'].iloc[-1]), 2),
                "previousClose": round(prev_close, 2),
                "change": round(change, 2),
                "pChange": round(pchange, 2),
                "marketCap": info.get("marketCap", 0),
                "volume": int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
                "source": "Yahoo Finance"
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "too many requests" in error_msg:
                if retry_count < 2:  # Retry up to 2 times for rate limiting
                    print(f"Rate limited for {symbol}, retrying in 3 seconds...")
                    time.sleep(3)
                    return fetch_single_stock(symbol, retry_count + 1)
            print(f"Error fetching {symbol}: {e}")
            return None
    
    # Use very conservative approach to avoid API rate limits - sequential processing
    stocks = []
    failed_stocks = []
    
    for i, symbol in enumerate(top_stocks):
        try:
            # Add delay between each request to avoid rate limiting
            if i > 0:
                time.sleep(0.5)  # 500ms delay between requests
            
            result = fetch_single_stock(symbol)
            if result:
                stocks.append(result)
            else:
                failed_stocks.append(symbol)
                
            if (i + 1) % 15 == 0:
                print(f"Progress: {len(stocks)} successful, {len(failed_stocks)} failed out of {i + 1} attempted")
                
        except Exception as e:
            print(f"Failed to process {symbol}: {e}")
            failed_stocks.append(symbol)
    
    end_time = time.time()
    print(f"Successfully fetched {len(stocks)} stocks in {end_time - start_time:.2f} seconds")
    if failed_stocks:
        print(f"Failed to fetch {len(failed_stocks)} stocks: {failed_stocks[:5]}{'...' if len(failed_stocks) > 5 else ''}")
    return stocks

def get_all_nse_stocks():
    """Alias for get_nifty_500_stocks for backward compatibility"""
    return get_nifty_500_stocks()
