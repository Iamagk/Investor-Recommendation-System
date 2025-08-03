#!/usr/bin/env python3
"""
Test script to verify the investment recommender is working after cleanup
"""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_stock_service():
    """Test the stock service"""
    print("🔍 Testing Stock Service...")
    try:
        from app.services.stock_service import get_nifty_500_stocks
        stocks = get_nifty_500_stocks()
        print(f"✅ Stock Service: Fetched {len(stocks)} stocks")
        
        if len(stocks) > 0:
            print(f"📊 Sample stock: {stocks[0]}")
            return True
        else:
            print("❌ No stocks fetched")
            return False
    except Exception as e:
        print(f"❌ Stock Service Error: {e}")
        return False

def test_mutual_funds_service():
    """Test the mutual funds service"""
    print("\n🔍 Testing Mutual Funds Service...")
    try:
        from app.services.mutual_funds_service import get_all_mutual_funds
        funds = get_all_mutual_funds()
        print(f"✅ Mutual Funds Service: Fetched {len(funds)} mutual funds")
        
        if len(funds) > 0:
            print(f"💰 Sample fund: {funds[0]}")
            return True
        else:
            print("❌ No mutual funds fetched")
            return False
    except Exception as e:
        print(f"❌ Mutual Funds Service Error: {e}")
        return False

def test_gold_service():
    """Test the gold service"""
    print("\n🔍 Testing Gold Service...")
    try:
        from data.fetch_gold import get_gold_rates
        gold_data = get_gold_rates()
        if not gold_data.empty:
            print(f"✅ Gold Service: Fetched {len(gold_data)} gold rates")
            print(f"🏆 Gold data sample:\n{gold_data.head()}")
            return True
        else:
            print("❌ No gold data fetched")
            return False
    except Exception as e:
        print(f"❌ Gold Service Error: {e}")
        return False

def test_api_imports():
    """Test that all API routes can be imported"""
    print("\n🔍 Testing API Route Imports...")
    try:
        from app.routes.stocks import router as stocks_router
        from app.routes.mutual_funds import router as mutual_funds_router
        from app.routes.gold import router as gold_router
        print("✅ All API routes imported successfully")
        return True
    except Exception as e:
        print(f"❌ API Import Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Investment Recommender - Post-Cleanup Testing")
    print("=" * 60)
    
    tests = [
        test_api_imports,
        test_stock_service,
        test_mutual_funds_service,
        test_gold_service
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📈 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your application is ready to run.")
        print("\n📝 To start the server:")
        print("   python start_server.py")
        print("\n🌐 Server will be available at:")
        print("   http://localhost:8000")
        print("   http://localhost:8000/docs (API documentation)")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 Try running the server anyway:")
        print("   python start_server.py")
    
    return passed == total

if __name__ == "__main__":
    main()
