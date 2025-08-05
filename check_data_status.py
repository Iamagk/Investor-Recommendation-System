#!/usr/bin/env python3
"""
Data Freshness Monitor - Check when investment data was last updated
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.db import get_postgres_connection
import pandas as pd

def check_data_freshness():
    """Check when each data source was last updated"""
    
    print("🔍 Investment Data Freshness Report")
    print("=" * 50)
    
    conn = get_postgres_connection()
    if not conn:
        print("❌ Could not connect to database")
        return
    
    current_time = datetime.now()
    print(f"📅 Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check stocks data
    try:
        stocks_data = pd.read_sql(
            "SELECT COUNT(*) as count, MAX(last_updated) as last_update FROM stocks", 
            conn
        )
        count = stocks_data.iloc[0]['count']
        last_update = stocks_data.iloc[0]['last_update']
        age = current_time - last_update if last_update else None
        
        status = "🟢 FRESH" if age and age.total_seconds() < 24*3600 else "🟡 OLD"
        print(f"📈 Stocks: {count} records")
        print(f"   Last updated: {last_update}")
        print(f"   Status: {status}")
        if age:
            print(f"   Age: {age.total_seconds()/3600:.1f} hours ago")
        print()
        
    except Exception as e:
        print(f"❌ Error checking stocks: {e}")
        print()
    
    # Check mutual funds data
    try:
        mf_data = pd.read_sql(
            "SELECT COUNT(*) as count, MAX(last_updated) as last_update FROM mutual_funds", 
            conn
        )
        count = mf_data.iloc[0]['count']
        last_update = mf_data.iloc[0]['last_update']
        age = current_time - last_update if last_update else None
        
        status = "🟢 FRESH" if age and age.total_seconds() < 24*3600 else "🟡 OLD"
        print(f"💰 Mutual Funds: {count} records")
        print(f"   Last updated: {last_update}")
        print(f"   Status: {status}")
        if age:
            print(f"   Age: {age.total_seconds()/3600:.1f} hours ago")
        print()
        
    except Exception as e:
        print(f"❌ Error checking mutual funds: {e}")
        print()
    
    # Check sector analysis data
    try:
        analysis_data = pd.read_sql(
            "SELECT COUNT(*) as count, MAX(analysis_date) as last_update FROM stock_sector_analysis", 
            conn
        )
        count = analysis_data.iloc[0]['count']
        last_update = analysis_data.iloc[0]['last_update']
        age = current_time - last_update if last_update else None
        
        status = "🟢 FRESH" if age and age.total_seconds() < 24*3600 else "🟡 OLD"
        print(f"📊 Sector Analysis: {count} records")
        print(f"   Last updated: {last_update}")
        print(f"   Status: {status}")
        if age:
            print(f"   Age: {age.total_seconds()/3600:.1f} hours ago")
        print()
        
    except Exception as e:
        print(f"❌ Error checking sector analysis: {e}")
        print()
    
    # Check comprehensive analysis
    try:
        comp_data = pd.read_sql(
            "SELECT COUNT(*) as count, MAX(analysis_date) as last_update FROM comprehensive_sector_analysis", 
            conn
        )
        count = comp_data.iloc[0]['count']
        last_update = comp_data.iloc[0]['last_update']
        age = current_time - last_update if last_update else None
        
        status = "🟢 FRESH" if age and age.total_seconds() < 24*3600 else "🟡 OLD"
        print(f"🔗 Comprehensive Analysis: {count} records")
        print(f"   Last updated: {last_update}")
        print(f"   Status: {status}")
        if age:
            print(f"   Age: {age.total_seconds()/3600:.1f} hours ago")
        print()
        
    except Exception as e:
        print(f"❌ Error checking comprehensive analysis: {e}")
        print()
    
    # Next scheduled scraper run
    next_9am = current_time.replace(hour=9, minute=0, second=0, microsecond=0)
    if current_time.hour >= 9:
        next_9am += timedelta(days=1)
    
    time_to_next = next_9am - current_time
    print(f"⏰ Next scheduled data update: {next_9am.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Time remaining: {time_to_next.total_seconds()/3600:.1f} hours")
    
    conn.close()

if __name__ == "__main__":
    check_data_freshness()
