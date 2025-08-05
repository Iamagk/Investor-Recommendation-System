#!/usr/bin/env python3
"""
API Health Monitor for Investment Recommender
Monitors critical endpoints to ensure they're working properly
"""

import requests
import time
import json

def test_endpoint(url, name, timeout=10):
    """Test an endpoint and return result"""
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # Convert to ms
        
        if response.status_code == 200:
            return {
                "name": name,
                "status": "✅ WORKING",
                "response_time": f"{response_time:.1f}ms",
                "status_code": response.status_code,
                "url": url
            }
        else:
            return {
                "name": name,
                "status": "❌ ERROR",
                "response_time": f"{response_time:.1f}ms",
                "status_code": response.status_code,
                "url": url
            }
    except requests.exceptions.Timeout:
        return {
            "name": name,
            "status": "⏰ TIMEOUT",
            "response_time": f">{timeout}s",
            "error": "Request timeout",
            "url": url
        }
    except requests.exceptions.ConnectionError:
        return {
            "name": name,
            "status": "🔌 CONNECTION ERROR",
            "error": "Cannot connect to server",
            "url": url
        }
    except Exception as e:
        return {
            "name": name,
            "status": "💥 EXCEPTION",
            "error": str(e),
            "url": url
        }

def monitor_api():
    """Monitor all critical API endpoints"""
    print("🔍 Investment Recommender API Health Check")
    print("=" * 50)
    
    endpoints = [
        ("http://localhost:8002/", "Server Root"),
        ("http://localhost:8002/ml/recommend/all/enhanced?top_n=2", "ML Predictions"),
        ("http://localhost:8002/recommend/comprehensive", "Comprehensive"),
        ("http://localhost:5173", "Frontend")
    ]
    
    results = []
    for url, name in endpoints:
        print(f"Testing {name}...")
        result = test_endpoint(url, name, timeout=8)
        results.append(result)
        
        status_emoji = result["status"].split()[0]
        time_info = result.get("response_time", "N/A")
        print(f"  {status_emoji} {name}: {time_info}")
        
        if "error" in result:
            print(f"    Error: {result['error']}")
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    
    working = sum(1 for r in results if "✅" in r["status"])
    total = len(results)
    
    print(f"Working endpoints: {working}/{total}")
    
    if working == total:
        print("🎉 All systems operational!")
    else:
        print("⚠️  Some endpoints need attention")
        for r in results:
            if "❌" in r["status"] or "⏰" in r["status"] or "🔌" in r["status"]:
                print(f"  - {r['name']}: {r['status']}")
    
    return results

if __name__ == "__main__":
    monitor_api()
