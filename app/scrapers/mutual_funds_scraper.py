import requests
from bs4 import BeautifulSoup
import time
from playwright.sync_api import sync_playwright
import asyncio

async def get_all_mutual_funds():
    """Async wrapper for mutual funds data fetching"""
    # Try to get real mutual funds data from AMFI
    try:
        return await asyncio.to_thread(get_mutual_funds_with_requests)
    except Exception as e:
        print(f"AMFI method failed: {e}")
        return {
            "error": f"Failed to fetch real mutual fund data: {str(e)}",
            "message": "AMFI API unavailable",
            "funds": []
        }

def get_mutual_funds_with_requests():
    """Try to get mutual funds data using AMFI"""
    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'no-cache'
    }
    
    # Try with a session and timeout
    session = requests.Session()
    response = session.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    
    lines = response.text.strip().split('\n')
    fund_data = []
    
    # Parse AMFI NAV data
    for line in lines:
        if ';' in line and len(line.split(';')) >= 5:
            parts = line.split(';')
            if len(parts) >= 5 and parts[4]:  # Check if NAV exists
                try:
                    scheme_name = parts[3].strip()
                    nav = float(parts[4])
                    if scheme_name and nav > 0:
                        fund_data.append({
                            'name': scheme_name,
                            'nav': nav,
                            'date': parts[5].strip('\r\n') if len(parts) > 5 else 'N/A'
                        })
                except (ValueError, IndexError):
                    continue
    
    return fund_data[:50]  # Return first 50 funds

def get_mutual_funds_with_playwright():
    """This method is kept for potential future use but not called in the main flow"""
    return []