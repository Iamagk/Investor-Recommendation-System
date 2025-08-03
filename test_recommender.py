#!/usr/bin/env python3
"""
Test script to debug the recommender output format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db import SessionLocal
from app.recommender.recommender import recommend_stocks

def test_recommender():
    """Test the recommender functions"""
    print("🔍 Testing Recommender Functions")
    print("=" * 50)
    
    try:
        db = SessionLocal()
        print("✅ Database connection established")
        
        # Test stock recommendations
        print("\n📈 Testing Stock Recommendations:")
        stocks = recommend_stocks(db, top_n=2)
        print(f"Type: {type(stocks)}")
        print(f"Length: {len(stocks) if stocks else 0}")
        
        if stocks:
            print("\nFirst recommendation structure:")
            print(f"Keys: {list(stocks[0].keys()) if isinstance(stocks[0], dict) else 'Not a dict'}")
            
            for i, stock in enumerate(stocks[:2]):
                print(f"\n{i+1}. Stock recommendation:")
                if isinstance(stock, dict):
                    for key, value in stock.items():
                        print(f"   {key}: {value}")
                else:
                    print(f"   Raw data: {stock}")
        else:
            print("   No stock recommendations returned")
        
        db.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_recommender()
