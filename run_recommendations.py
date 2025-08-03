#!/usr/bin/env python3
"""
Simple script to run recommendation functions directly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db import get_postgres_connection
from app.recommender.recommender import (
    recommend_assets,
    full_recommendation_with_budget,
    recommend_stocks,
    recommend_mutual_funds,
    recommend_gold
)

def main():
    """Run recommendation functions"""
    print("🎯 Investment Recommendation System")
    print("=" * 50)
    
    # Get database connection
    try:
        db = next(get_postgres_connection())
        print("✅ Database connection established")
    except Exception as e:
        try:
            # Alternative connection method
            from app.db import SessionLocal
            db = SessionLocal()
            print("✅ Database connection established (alternative method)")
        except Exception as e2:
            print(f"❌ Database connection failed: {e}")
            print(f"❌ Alternative method also failed: {e2}")
            print("📝 Note: Make sure PostgreSQL is running and configured properly")
            return
    
    # Test individual recommendations
    print("\n📈 Stock Recommendations:")
    try:
        stocks = recommend_stocks(db, top_n=3)
        for i, stock in enumerate(stocks[:3], 1):
            symbol = stock.get('symbol', stock.get('sector', 'N/A'))
            score = stock.get('predicted_roi', stock.get('score', 0))
            print(f"   {i}. {symbol} - Score: {score}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n🏦 Mutual Fund Recommendations:")
    try:
        mfs = recommend_mutual_funds(db, top_n=3)
        for i, mf in enumerate(mfs[:3], 1):
            sector = mf.get('sector', 'N/A')
            score = mf.get('score', 0)
            print(f"   {i}. {sector} - Score: {score}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n🥇 Gold Recommendation:")
    try:
        gold = recommend_gold(db)
        if gold:
            asset_type = gold.get('asset_type', 'Gold')
            score = gold.get('score', 0)
            print(f"   1. {asset_type} - Score: {score}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test comprehensive recommendations
    print("\n🎯 Comprehensive Asset Recommendations:")
    try:
        all_assets = recommend_assets(db, top_n=3, use_ml=True)
        if all_assets.get('status') == 'success':
            recs = all_assets.get('recommendations', {})
            total = all_assets.get('total_recommendations', 0)
            print(f"   Total recommendations: {total}")
            print(f"   ML Enhanced: {all_assets.get('ml_enhanced', False)}")
        else:
            print(f"   Error: {all_assets.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test budget-based recommendations
    print("\n💰 Budget-Based Recommendations (₹1,00,000):")
    try:
        budget_recs = full_recommendation_with_budget(
            db, 
            budget=100000, 
            top_n=3, 
            use_ml=True, 
            risk_profile="balanced"
        )
        if budget_recs.get('status') == 'success':
            portfolio = budget_recs.get('optimized_portfolio', {})
            total_allocated = portfolio.get('total_allocated', 0)
            remaining = portfolio.get('remaining_budget', 0)
            print(f"   Total allocated: ₹{total_allocated:,.2f}")
            print(f"   Remaining budget: ₹{remaining:,.2f}")
            print(f"   Risk profile: {budget_recs.get('risk_profile', 'N/A')}")
            print(f"   Market condition: {budget_recs.get('market_analysis', {}).get('condition', 'N/A')}")
        else:
            print(f"   Error: {budget_recs.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ Recommendation demo completed!")
    
    # Close database connection
    try:
        db.close()
    except:
        pass

if __name__ == "__main__":
    main()
