import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.scrapers.mutual_funds_scraper import get_all_mutual_funds
from app.scrapers.stocks_scraper import get_active_nse_stocks

print("\n--- MUTUAL FUNDS ---")
funds = get_all_mutual_funds()
for fund in funds[:5]:
    print(fund)

print("\n--- STOCKS ---")
stocks = get_active_nse_stocks()
for stock in stocks[:5]:
    print(stock)