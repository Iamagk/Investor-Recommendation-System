#!/usr/bin/env python3
"""
Demo script to run the ML predictor functions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db import get_postgres_connection
from app.ml.predictor import (
    recommend_all_assets_enhanced,
    get_model_info,
    train_and_save_ensemble_models,
    recommend_stocks_enhanced,
    recommend_mutual_funds_enhanced,
    recommend_gold_enhanced
)

def main():
    """Main function to demonstrate predictor functionality"""
    print("🚀 Starting ML Predictor Demo")
    print("=" * 50)
    
    # Get database connection
    try:
        db = next(get_postgres_connection())
        print("✅ Database connection established")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("📝 Note: Make sure your database is running and configured properly")
        return
    
    # Check model status
    print("\n📊 Model Information:")
    model_info = get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Train models if not available
    if not model_info.get('rf_model_available', False):
        print("\n🔧 Training ensemble models...")
        try:
            training_result = train_and_save_ensemble_models(db, "stocks")
            print(f"   Training result: {training_result}")
        except Exception as e:
            print(f"   Training failed: {e}")
    
    # Test stock recommendations
    print("\n📈 Getting Stock Recommendations:")
    try:
        stock_recs = recommend_stocks_enhanced(db, top_n=3)
        print(f"   Found {len(stock_recs.get('recommendations', []))} stock recommendations")
        for i, stock in enumerate(stock_recs.get('recommendations', [])[:3], 1):
            print(f"   {i}. {stock.get('symbol', 'N/A')} - Expected Return: {stock.get('predicted_return', 0):.2f}%")
    except Exception as e:
        print(f"   ❌ Stock recommendations failed: {e}")
    
    # Test mutual fund recommendations
    print("\n🏦 Getting Mutual Fund Recommendations:")
    try:
        mf_recs = recommend_mutual_funds_enhanced(db, top_n=3)
        print(f"   Found {len(mf_recs.get('recommendations', []))} mutual fund recommendations")
        for i, mf in enumerate(mf_recs.get('recommendations', [])[:3], 1):
            print(f"   {i}. {mf.get('scheme_name', 'N/A')[:50]}... - Expected Return: {mf.get('predicted_return', 0):.2f}%")
    except Exception as e:
        print(f"   ❌ Mutual fund recommendations failed: {e}")
    
    # Test gold recommendations
    print("\n🥇 Getting Gold Recommendations:")
    try:
        gold_recs = recommend_gold_enhanced(db, top_n=2)
        print(f"   Found {len(gold_recs.get('recommendations', []))} gold recommendations")
        for i, gold in enumerate(gold_recs.get('recommendations', [])[:2], 1):
            print(f"   {i}. {gold.get('state', 'N/A')} - Price: ₹{gold.get('price_24k', 0):.2f}/gram")
    except Exception as e:
        print(f"   ❌ Gold recommendations failed: {e}")
    
    # Test comprehensive recommendations
    print("\n🎯 Getting Comprehensive Asset Recommendations:")
    try:
        all_recs = recommend_all_assets_enhanced(db, stocks_n=2, mf_n=2, gold_n=1)
        print("   Summary:")
        print(f"   - Total recommendations: {all_recs.get('total_recommendations', 0)}")
        print(f"   - Average expected return: {all_recs.get('average_expected_return', 0):.2f}%")
        print(f"   - Risk score: {all_recs.get('risk_score', 0):.2f}")
        
        # Show breakdown
        for asset_type in ['stocks', 'mutual_funds', 'gold']:
            if asset_type in all_recs:
                count = len(all_recs[asset_type].get('recommendations', []))
                avg_return = all_recs[asset_type].get('average_expected_return', 0)
                print(f"   - {asset_type.title()}: {count} items, avg return: {avg_return:.2f}%")
                
    except Exception as e:
        print(f"   ❌ Comprehensive recommendations failed: {e}")
    
    print("\n✅ ML Predictor Demo completed!")
    print("=" * 50)
    
    # Close database connection
    try:
        db.close()
        print("📝 Database connection closed")
    except:
        pass

if __name__ == "__main__":
    main()
