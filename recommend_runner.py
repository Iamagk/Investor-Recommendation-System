#!/usr/bin/env python3
"""
Command-line recommendation runner with formatted output
Similar to backtest_runner.py but for real-time recommendations
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import get_db
from app.recommender.recommender import recommend_assets, recommend_stocks, recommend_mutual_funds, recommend_gold
from datetime import datetime


def print_formatted_recommendations(top_n=5, use_ml=True):
    """Print formatted recommendations to console"""
    
    # Get database session
    db = next(get_db())
    
    try:
        # Get recommendations
        result = recommend_assets(db, top_n=top_n, use_ml=use_ml, use_realtime=False)
        
        if result["status"] != "success":
            print(f"❌ Error: {result.get('message', 'Unknown error')}")
            return
        
        # Print formatted output
        print("=" * 80)
        print("🎯 SMART INVESTMENT RECOMMENDATIONS")
        print("=" * 80)
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🤖 ML Enhanced: {'Yes' if use_ml else 'No'}")
        print(f"📊 Total Assets Analyzed: {result.get('total_recommendations', 0)}")
        print()
        
        # Print stocks
        stocks = result["recommendations"].get("stocks", [])
        if stocks:
            print("📈 TOP STOCK RECOMMENDATIONS:")
            print("-" * 50)
            for i, stock in enumerate(stocks[:top_n], 1):
                print(f"{i}. {stock.get('company_name', stock.get('symbol', 'Unknown'))}")
                print(f"   Symbol: {stock.get('symbol', 'N/A')}")
                print(f"   Sector: {stock.get('sector', 'N/A')}")
                print(f"   Predicted ROI: {stock.get('predicted_roi', 0):.2f}%")
                print(f"   Current Price: ₹{stock.get('current_price', 0):,.2f}")
                print(f"   Signal: {stock.get('signal', 'HOLD')}")
                print()
        
        # Print mutual funds
        mfs = result["recommendations"].get("mutual_funds", [])
        if mfs:
            print("🏦 TOP MUTUAL FUND SECTORS:")
            print("-" * 50)
            for i, mf in enumerate(mfs[:top_n], 1):
                print(f"{i}. {mf.get('sector', 'Unknown')} Sector")
                print(f"   Score: {mf.get('score', 0):.2f}")
                print(f"   Return %: {mf.get('return_percentage', 0):.2f}%")
                print(f"   Volatility: {mf.get('volatility', 0):.2f}")
                print(f"   Signal: {mf.get('signal', 'HOLD')}")
                print()
        
        # Print gold
        gold = result["recommendations"].get("gold", {})
        if gold:
            print("🥇 GOLD INVESTMENT:")
            print("-" * 50)
            print(f"   Asset Type: {gold.get('asset_type', 'Gold')}")
            print(f"   Score: {gold.get('score', 0):.2f}")
            print(f"   Signal: {gold.get('signal', 'HOLD')}")
            print()
        
        # Print summary
        print("=" * 80)
        print("📋 INVESTMENT SUMMARY:")
        print("=" * 80)
        if stocks:
            best_stock = stocks[0]
            print(f"🥇 Best Stock: {best_stock.get('company_name', 'Unknown')} (ROI: {best_stock.get('predicted_roi', 0):.2f}%)")
        if mfs:
            best_mf = mfs[0]
            print(f"🏆 Best Sector: {best_mf.get('sector', 'Unknown')} (Score: {best_mf.get('score', 0):.2f})")
        
        print()
        print("💡 Note: This is AI-generated advice. Please consult a financial advisor.")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")
    finally:
        db.close()


def print_market_analysis():
    """Print market condition analysis"""
    from app.recommender.recommender import detect_market_condition, get_dynamic_weights
    
    db = next(get_db())
    try:
        market_info = detect_market_condition(db)
        weights = get_dynamic_weights(market_condition=market_info["condition"])
        
        print("=" * 80)
        print("📊 MARKET ANALYSIS")
        print("=" * 80)
        print(f"Market Condition: {market_info['condition'].upper()}")
        print(f"Volatility Index: {market_info['volatility_index']:.1f}")
        print(f"Average Return: {market_info['avg_return']:.2f}%")
        print(f"Average Volatility: {market_info['avg_volatility']:.2f}")
        print()
        print("Dynamic Weights Applied:")
        print(f"  Return Weight: {weights['return']:.2f}")
        print(f"  Volatility Penalty: {weights['volatility']:.2f}")
        print(f"  Momentum Factor: {weights['momentum']:.2f}")
        print("=" * 80)
        print()
        
    except Exception as e:
        print(f"❌ Error in market analysis: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate formatted investment recommendations")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top recommendations")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML predictions")
    parser.add_argument("--market-only", action="store_true", help="Show only market analysis")
    
    args = parser.parse_args()
    
    if args.market_only:
        print_market_analysis()
    else:
        print_market_analysis()
        print_formatted_recommendations(top_n=args.top_n, use_ml=not args.no_ml)
