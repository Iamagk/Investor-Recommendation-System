#!/usr/bin/env python3
import yfinance as yf
import pandas as pd

def test_single_stock():
    symbol = "RELIANCE.NS"
    print(f"Testing {symbol}...")
    
    try:
        # Fetch data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        info = ticker.info
        
        print(f"History data shape: {hist.shape}")
        print(f"History columns: {hist.columns.tolist()}")
        print(f"Info keys: {list(info.keys())}")
        
        # Check what we get
        print(f"Current price: {hist['Close'].iloc[-1]}")
        print(f"Moving average 20: {hist['Close'].rolling(20).mean().iloc[-1]}")
        
        # Calculate RSI
        close_prices = hist['Close']
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        print(f"RSI: {rsi.iloc[-1]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_stock()
