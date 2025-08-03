# app/utils/realtime_prices.py

import yfinance as yf
from functools import lru_cache
from datetime import datetime, timedelta
import time

CACHE = {}
CACHE_TTL = 60  # seconds

def _cache_get(ticker):
    entry = CACHE.get(ticker)
    if entry:
        value, timestamp = entry
        if time.time() - timestamp < CACHE_TTL:
            return value
    return None

def _cache_set(ticker, value):
    CACHE[ticker] = (value, time.time())

def fetch_realtime_price(ticker: str) -> float:
    """Fetch real-time price with simple in-memory cache and error handling."""
    cached_price = _cache_get(ticker)
    if cached_price is not None:
        return cached_price

    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d", interval="1m")
        if not data.empty:
            price = data['Close'].iloc[-1]
            _cache_set(ticker, price)
            return float(price)
        else:
            raise ValueError(f"No data for ticker: {ticker}")
    except Exception as e:
        print(f"[Error] Failed to fetch real-time price for {ticker}: {e}")
        return -1.0  # Default or error value