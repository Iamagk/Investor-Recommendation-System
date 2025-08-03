# data/fetch_gold.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
from datetime import datetime
import time

def get_gold_rates():
    """Get gold rates with multiple reliable sources and robust error handling"""
    sources = [
        ("Yahoo Finance Gold ETFs", get_gold_rates_from_yfinance),
        ("MoneyControl", get_gold_rates_from_moneycontrol),
        ("GoodReturns", get_gold_rates_from_goodreturns),
        ("Fallback Gold Data", get_fallback_gold_data)
    ]
    
    for source_name, source_func in sources:
        try:
            print(f"Trying {source_name}...")
            data = source_func()
            if data is not None and not data.empty:
                print(f"✅ Successfully fetched gold data from {source_name}")
                return data
        except Exception as e:
            print(f"❌ {source_name} failed: {e}")
            continue
    
    print("⚠️ All gold data sources failed, returning empty DataFrame")
    return pd.DataFrame(columns=['Date', 'Price_per_gram', 'Source'])

def get_gold_rates_from_goodreturns():
    """Enhanced GoodReturns scraper with multiple table search strategies"""
    url = "https://www.goodreturns.in/gold-rates/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Multiple table search strategies
        table_selectors = [
            "table.gold_silver_table",
            "table[class*='gold']",
            "table[class*='rate']",
            ".gold-rates-table",
            "#gold-rates-table",
            "table:contains('Gold')",
            "table"  # Last resort - any table
        ]
        
        for selector in table_selectors:
            try:
                if selector == "table":
                    tables = soup.find_all("table")
                    for table in tables:
                        if table and ("gold" in str(table).lower() or "rate" in str(table).lower()):
                            data = extract_gold_data_from_table(table)
                            if data:
                                return data
                else:
                    table = soup.select_one(selector)
                    if table:
                        data = extract_gold_data_from_table(table)
                        if data:
                            return data
            except Exception:
                continue
        
        raise Exception("No valid gold rates table found on GoodReturns")
        
    except requests.RequestException as e:
        raise Exception(f"Network error accessing GoodReturns: {str(e)}")
    except Exception as e:
        raise Exception(f"GoodReturns parsing error: {str(e)}")

def extract_gold_data_from_table(table):
    """Extract gold data from a table element"""
    try:
        rows = table.find_all("tr")
        if len(rows) < 2:  # Need at least header + one data row
            return None
            
        data = []
        for row in rows[1:]:  # Skip header
            cols = row.find_all(["td", "th"])
            if len(cols) >= 2:
                try:
                    # Try to extract city and rate information
                    city_text = cols[0].get_text(strip=True)
                    if not city_text or city_text.lower() in ['city', 'location', 'place']:
                        continue
                        
                    # Look for price columns
                    for i, col in enumerate(cols[1:], 1):
                        price_text = col.get_text(strip=True)
                        price_text = price_text.replace("₹", "").replace(",", "").replace("Rs", "")
                        
                        try:
                            price = float(price_text)
                            if 3000 <= price <= 10000:  # Reasonable gold price range per gram in INR
                                data.append({
                                    "City": city_text,
                                    "Price_per_gram": price,
                                    "Purity": f"Column_{i}",
                                    "Source": "GoodReturns",
                                    "Date": datetime.now().strftime("%Y-%m-%d")
                                })
                                break
                        except (ValueError, TypeError):
                            continue
                except Exception:
                    continue
        
        if data:
            return pd.DataFrame(data)
        return None
        
    except Exception:
        return None

def get_gold_rates_from_yfinance():
    """Enhanced YFinance method for gold ETF data"""
    try:
        # Comprehensive list of gold-related symbols
        gold_symbols = {
            'GLD': 'SPDR Gold Trust',
            'GOLDBEES.NS': 'Goldman Sachs Gold BEeS',
            'GOLDIETF.NS': 'ICICI Prudential Gold ETF',
            'SGLD': 'Aberdeen Standard Gold ETF',
            'GC=F': 'Gold Futures'
        }
        
        data = []
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        for symbol, name in gold_symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")  # Get more data for reliability
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    
                    # Convert to INR per gram approximation for Indian ETFs
                    if symbol.endswith('.NS'):
                        # Indian ETFs - already in INR, convert to per gram
                        price_per_gram = round(float(current_price), 2)
                    else:
                        # International - convert USD to INR and calculate per gram
                        # Approximate conversion: 1 oz = 31.1035 grams, USD to INR ≈ 83
                        price_per_gram = round(float(current_price) * 83 / 31.1035, 2)
                    
                    data.append({
                        "Symbol": symbol,
                        "Name": name,
                        "Price_per_gram": price_per_gram,
                        "Date": current_date,
                        "Source": "Yahoo Finance"
                    })
                    
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue
        
        if not data:
            raise Exception("No gold ETF data available from Yahoo Finance")
            
        return pd.DataFrame(data)
        
    except Exception as e:
        raise Exception(f"YFinance gold data error: {str(e)}")

def get_gold_rates_from_moneycontrol():
    """Scrape gold rates from MoneyControl"""
    try:
        url = "https://www.moneycontrol.com/commodity/gold-price.html"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Look for gold price elements
        price_elements = soup.find_all(["span", "div"], class_=lambda x: x and "price" in x.lower())
        
        for element in price_elements:
            text = element.get_text(strip=True)
            if "₹" in text or "Rs" in text:
                try:
                    price_str = text.replace("₹", "").replace("Rs", "").replace(",", "").strip()
                    price = float(price_str)
                    
                    if 3000 <= price <= 10000:  # Reasonable range for gold price per gram
                        return pd.DataFrame([{
                            "Source": "MoneyControl",
                            "Price_per_gram": price,
                            "Date": datetime.now().strftime("%Y-%m-%d"),
                            "Currency": "INR"
                        }])
                except (ValueError, TypeError):
                    continue
        
        raise Exception("No valid gold price found on MoneyControl")
        
    except Exception as e:
        raise Exception(f"MoneyControl scraping error: {str(e)}")

def get_fallback_gold_data():
    """Fallback method using Indian gold market API or estimated data"""
    try:
        # Try to get approximate gold price using currency conversion
        import requests
        
        # Get current gold price in USD from a free API
        try:
            response = requests.get("https://api.metals.live/v1/spot/gold", timeout=10)
            if response.status_code == 200:
                data = response.json()
                usd_price_per_oz = float(data['price'])
                
                # Convert to INR per gram (approximate USD to INR = 83)
                inr_price_per_gram = round((usd_price_per_oz * 83) / 31.1035, 2)
                
                return pd.DataFrame([{
                    "Source": "Fallback (metals.live API)",
                    "Price_per_gram": inr_price_per_gram,
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                    "Currency": "INR",
                    "Note": "Converted from USD spot price"
                }])
        except:
            pass
        
        # If API fails, use reasonable estimated range based on current market
        estimated_price = 6500  # Approximate current gold price per gram in INR
        
        return pd.DataFrame([{
            "Source": "Fallback (Estimated)",
            "Price_per_gram": estimated_price,
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Currency": "INR",
            "Note": "Estimated price when all sources fail"
        }])
        
    except Exception as e:
        raise Exception(f"Fallback method failed: {str(e)}")