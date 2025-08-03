import requests
import yfinance as yf
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time

def get_active_nse_stocks():
    """Get active NSE stocks using Yahoo Finance"""
    try:
        return get_stocks_with_yfinance()
    except Exception as e:
        print(f"YFinance method failed: {e}")
        return {
            "error": f"Failed to fetch real stock data: {str(e)}",
            "message": "Yahoo Finance API unavailable",
            "stocks": []
        }

def get_stocks_with_yfinance():
    """Get stock data using YFinance library (more reliable)"""
    # Popular NSE stocks (reduced to avoid rate limiting)
    nse_symbols = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
        'ASIANPAINT.NS', 'MARUTI.NS', 'AXISBANK.NS', 'LT.NS', 'NESTLEIND.NS',
        'TITAN.NS', 'WIPRO.NS', 'BAJFINANCE.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS'
    ]
    
    stock_data = []
    
    for symbol in nse_symbols:
        try:
            # Add delay to avoid rate limiting
            time.sleep(0.3)
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="2d")
            
            if not hist.empty and len(hist) >= 1:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[0] if len(hist) > 1 else current_price
                change = current_price - prev_close
                change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
                
                stock_data.append({
                    'symbol': symbol.replace('.NS', ''),
                    'name': info.get('longName', symbol.replace('.NS', '')),
                    'ltp': round(current_price, 2),
                    'change': round(change, 2),
                    'change_percent': round(change_percent, 2),
                    'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 'N/A'
                })
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            continue
    
    return stock_data

def get_stocks_with_playwright():
    """Original playwright method - kept for potential future use"""
    url = "https://www.moneycontrol.com/stocks/marketstats/nse-mostactive-stocks/index.php"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        page = browser.new_page()
        
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        
        try:
            page.goto(url, timeout=60000)
            time.sleep(3)
            
            # Try multiple selectors
            try:
                page.wait_for_selector('table', timeout=10000)
            except:
                try:
                    page.wait_for_selector('.tbldata14', timeout=5000)
                except:
                    print("No table found, proceeding with current content")
            
            content = page.content()
        except Exception as e:
            print(f"Error loading page: {e}")
            return []
        finally:
            browser.close()

    soup = BeautifulSoup(content, 'html.parser')
    stock_data = []

    table = soup.find('table', {'class': 'tbldata14'})
    if table:
        rows = table.find_all('tr')[1:]  # Skip the table header
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 5:
                name = cols[0].get_text(strip=True)
                ltp = cols[1].get_text(strip=True)
                chg = cols[2].get_text(strip=True)
                chg_percent = cols[3].get_text(strip=True)
                vol = cols[4].get_text(strip=True)

                stock_data.append({
                    'name': name,
                    'ltp': ltp,
                    'change': chg,
                    'change_percent': chg_percent,
                    'volume': vol
                })

    return stock_data