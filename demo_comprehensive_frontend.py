#!/usr/bin/env python3
"""
Demo script to showcase the comprehensive investment recommendations in the frontend
"""

import requests
import json
from datetime import datetime

def test_comprehensive_api():
    """Test the comprehensive recommendations API"""
    
    url = "http://localhost:8000/api/portfolio/predict"
    
    # Sample request data
    test_data = {
        "amount": 100000,
        "risk_tolerance": 6,  # Balanced risk
        "expected_return": 12,
        "selected_options": ["stocks", "mutualFunds", "gold"]
    }
    
    print("🚀 Testing Comprehensive Investment Recommendations")
    print("=" * 60)
    print(f"Request: {json.dumps(test_data, indent=2)}")
    print("=" * 60)
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ API Response Successful!")
            print(f"Status Code: {response.status_code}")
            print()
            
            # Show allocation
            allocation = data.get("recommended_allocation", {})
            print("📊 RECOMMENDED ALLOCATION:")
            print(f"  Stocks: {allocation.get('stocks', 0)}%")
            print(f"  Mutual Funds: {allocation.get('mutualFunds', 0)}%")
            print(f"  Gold: {allocation.get('gold', 0)}%")
            print()
            
            # Show expected return
            print(f"📈 EXPECTED RETURN: {data.get('expected_return', 0)}%")
            print()
            
            # Show comprehensive data availability
            detailed = data.get("detailed_investments", {})
            if detailed:
                print("🎯 COMPREHENSIVE DATA AVAILABLE:")
                
                if "stocks" in detailed and detailed["stocks"].get("comprehensive_data"):
                    stocks_data = detailed["stocks"]["comprehensive_data"]
                    print(f"  📊 Stocks: {len(stocks_data)} sectors analyzed")
                    for i, sector in enumerate(stocks_data[:2], 1):
                        stock_count = len(sector.get("stocks", []))
                        print(f"    {i}. {sector['sector']}: {stock_count} stocks, {sector['predicted_return']:.1f}% return")
                
                if "mutualFunds" in detailed and detailed["mutualFunds"].get("comprehensive_data"):
                    mf_data = detailed["mutualFunds"]["comprehensive_data"]
                    print(f"  💼 Mutual Funds: {len(mf_data)} sectors analyzed")
                    for i, sector in enumerate(mf_data[:2], 1):
                        fund_count = len(sector.get("funds", []))
                        print(f"    {i}. {sector['sector']}: {fund_count} funds, {sector['predicted_return']:.1f}% return")
                
                if "gold" in detailed and detailed["gold"].get("comprehensive_data"):
                    gold_data = detailed["gold"]["comprehensive_data"]
                    print(f"  🥇 Gold: {len(gold_data)} options analyzed")
                    for i, sector in enumerate(gold_data[:2], 1):
                        investment_count = len(sector.get("gold_investments", []))
                        print(f"    {i}. {sector['sector']}: {investment_count} investments, {sector['predicted_return']:.1f}% return")
            
            print()
            print("🌐 FRONTEND INTEGRATION:")
            print("  Frontend URL: http://localhost:5175")
            print("  Features Available:")
            print("    ✅ Comprehensive sector analysis")
            print("    ✅ Individual stock recommendations with timing")
            print("    ✅ Mutual fund strategies")
            print("    ✅ Gold investment options")
            print("    ✅ Detailed investment commentary")
            print("    ✅ ML-based predictions")
            print()
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        print("Make sure the backend server is running on http://localhost:8000")

def show_usage_instructions():
    """Show how to use the comprehensive recommendations"""
    
    print()
    print("📋 HOW TO USE COMPREHENSIVE RECOMMENDATIONS:")
    print("=" * 60)
    print("1. Open your browser to: http://localhost:5175")
    print("2. Fill in your investment details:")
    print("   - Investment Amount (e.g., ₹100,000)")
    print("   - Risk Appetite (Low/Medium/High)")
    print("   - Select asset classes (Stocks, Mutual Funds, Gold)")
    print("3. Click 'Get Recommendations'")
    print("4. View the comprehensive analysis including:")
    print("   - Basic allocation breakdown")
    print("   - Detailed sector analysis")
    print("   - Individual stock recommendations")
    print("   - Timing analysis with entry/exit dates")
    print("   - Investment strategies")
    print("   - ML confidence scores")
    print()
    print("🎯 NEW FEATURES:")
    print("  • Backtest-quality detailed analysis")
    print("  • Sector-wise recommendations")
    print("  • Individual stock timing analysis")
    print("  • Professional investment strategies")
    print("  • Real-time ML predictions")
    print("  • Comprehensive portfolio insights")

if __name__ == "__main__":
    print(f"📅 Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test the API
    test_comprehensive_api()
    
    # Show usage instructions
    show_usage_instructions()
    
    print()
    print("🎉 Comprehensive Investment Recommendations are ready!")
    print("The frontend now displays detailed analysis like the backtest results.")
