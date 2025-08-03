import requests
import asyncio
from app.utils.mutual_fund_fetcher import scrape_all_mutual_funds

AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"

def get_mutual_fund_data():
    try:
        response = requests.get(AMFI_URL)
        raw_lines = response.text.splitlines()
        
        funds = []
        for line in raw_lines:
            if ";" in line:
                parts = line.split(";")
                if len(parts) >= 6 and parts[0].strip().isdigit():
                    fund = {
                        "scheme_code": parts[0].strip(),
                        "isin_div_payout": parts[1],
                        "isin_div_reinvestment": parts[2],
                        "scheme_name": parts[3].strip(),
                        "nav": float(parts[4]) if parts[4] and parts[4].strip() and parts[4].strip().lower() not in ['n.a.', '-', ''] else None,
                        "nav_date": parts[5].strip() if len(parts) > 5 else None,
                        # Map to our database fields
                        "fund_name": parts[3].strip(),
                        "fund_house": "Unknown",  # We'll extract this from scheme_name if needed
                        "category": "General",
                        "returns_1y": None,
                        "returns_3y": None,
                        "risk_level": "Medium"
                    }
                    funds.append(fund)

        return funds
    except Exception as e:
        return {"error": str(e)}

def get_all_mutual_funds():
    """Get all mutual funds using the async scraper utility"""
    return asyncio.run(scrape_all_mutual_funds())