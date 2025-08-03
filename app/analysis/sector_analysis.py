# analysis/sector_analysis.py

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pandas as pd
from datetime import datetime, timedelta
from app.db import get_postgres_connection
from sqlalchemy import create_engine

# Number of days to consider for return calculation
LOOKBACK_DAYS = 30

# Create SQLAlchemy engine for pandas operations
try:
    # Build DATABASE_URL from the connection parameters in db.py
    DATABASE_URL = "postgresql://postgres:pass@localhost:5432/postgres"
    engine = create_engine(DATABASE_URL)
except:
    # Fallback if SQLAlchemy not available
    engine = None

def fetch_data():
    """Fetch stock data from the database for sector analysis"""
    conn = get_postgres_connection()
    if not conn:
        print("Failed to connect to database")
        return pd.DataFrame()
    
    try:
        # Query the actual stocks table with available columns
        query = """
            SELECT symbol, name, sector, last_price, change_percent, last_updated
            FROM stocks
            WHERE last_updated >= %s
            ORDER BY last_updated ASC
        """
        start_date = datetime.now() - timedelta(days=30)  # Use 30 days for recent data
        df = pd.read_sql(query, conn, params=(start_date,))
        return df
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def fetch_mutual_fund_data():
    """Fetch mutual fund data from the database"""
    conn = get_postgres_connection()
    if not conn:
        print("Failed to connect to database")
        return pd.DataFrame()
    
    try:
        # Query the mutual_funds table
        query = """
            SELECT fund_name AS symbol, fund_name AS name, category AS sector, 
            nav AS last_price, returns_1y AS change_percent, last_updated
            FROM mutual_funds
            WHERE last_updated >= %s AND nav > 0
            ORDER BY last_updated ASC
        """
        start_date = datetime.now() - timedelta(days=30)
        df = pd.read_sql(query, conn, params=(start_date,))
        return df
    except Exception as e:
        print(f"Error fetching mutual fund data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def fetch_gold_data():
    """Fetch gold data from the database"""
    conn = get_postgres_connection()
    if not conn:
        print("Failed to connect to database")
        return pd.DataFrame()
    
    try:
        # Query the gold_prices table
        query = """
            SELECT 'Gold' AS symbol, 'Gold Investment' AS name, 'Precious Metals' AS sector,
            price_per_gram AS last_price, 0 AS change_percent, updated_at AS last_updated
            FROM gold_prices
            WHERE updated_at >= %s
            ORDER BY updated_at ASC
        """
        start_date = datetime.now() - timedelta(days=30)
        df = pd.read_sql(query, conn, params=(start_date,))
        return df
    except Exception as e:
        print(f"Error fetching gold data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def calculate_sector_scores():
    """Enhanced sector scoring function using SQL and pandas"""
    today = datetime.now().date()
    start_date = today - timedelta(days=LOOKBACK_DAYS)
    
    print(f"📊 Calculating sector scores for last {LOOKBACK_DAYS} days (from {start_date})")
    
    # Load data from database using actual table names
    try:
        if engine:
            # Use SQLAlchemy engine for better pandas integration
            stocks_df = pd.read_sql(f"""
                SELECT symbol, name, sector, last_price as close, change_percent, last_updated as date
                FROM stocks 
                WHERE last_updated >= '{start_date}'
                ORDER BY last_updated
            """, engine)
            
            mutual_df = pd.read_sql(f"""
                SELECT fund_name as symbol, category as sector, nav as close, 
                returns_1y as change_percent, last_updated as date
                FROM mutual_funds 
                WHERE last_updated >= '{start_date}' AND nav > 0
                ORDER BY last_updated
            """, engine)
            
            gold_df = pd.read_sql(f"""
                SELECT 'Gold' as symbol, 'Precious Metals' as sector, 
                price_per_gram as price, updated_at as date
                FROM gold_prices 
                WHERE updated_at >= '{start_date}'
                ORDER BY updated_at
            """, engine)
        else:
            # Fallback to manual connection
            stocks_df = fetch_stocks_for_scoring(start_date)
            mutual_df = fetch_mutual_funds_for_scoring(start_date)
            gold_df = fetch_gold_for_scoring(start_date)
            
    except Exception as e:
        print(f"Error loading data: {e}")
        stocks_df = pd.DataFrame()
        mutual_df = pd.DataFrame()
        gold_df = pd.DataFrame()
    
    print(f"   📈 Stocks: {len(stocks_df)} records")
    print(f"   💰 Mutual Funds: {len(mutual_df)} records")
    print(f"   🏆 Gold: {len(gold_df)} records")
    
    # Process each asset type
    stock_sector_scores = process_asset_sector_scores(stocks_df, asset_type="stock") if not stocks_df.empty else pd.DataFrame()
    mutual_sector_scores = process_asset_sector_scores(mutual_df, asset_type="mutual_fund") if not mutual_df.empty else pd.DataFrame()
    gold_sector_scores = process_gold_score(gold_df) if not gold_df.empty else pd.DataFrame()

    return {
        "stocks": stock_sector_scores.to_dict(orient="records") if not stock_sector_scores.empty else [],
        "mutual_funds": mutual_sector_scores.to_dict(orient="records") if not mutual_sector_scores.empty else [],
        "gold": gold_sector_scores.to_dict(orient="records") if not gold_sector_scores.empty else []
    }

def fetch_stocks_for_scoring(start_date):
    """Fallback function to fetch stocks data"""
    conn = get_postgres_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
            SELECT symbol, name, sector, last_price as close, change_percent, last_updated as date
            FROM stocks 
            WHERE last_updated >= %s
            ORDER BY last_updated
        """
        return pd.read_sql(query, conn, params=(start_date,))
    except Exception as e:
        print(f"Error fetching stocks: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def fetch_mutual_funds_for_scoring(start_date):
    """Fallback function to fetch mutual funds data"""
    conn = get_postgres_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
            SELECT fund_name as symbol, category as sector, nav as close, 
            returns_1y as change_percent, last_updated as date
            FROM mutual_funds 
            WHERE last_updated >= %s AND nav > 0
            ORDER BY last_updated
        """
        return pd.read_sql(query, conn, params=(start_date,))
    except Exception as e:
        print(f"Error fetching mutual funds: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def fetch_gold_for_scoring(start_date):
    """Fallback function to fetch gold data"""
    conn = get_postgres_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
            SELECT 'Gold' as symbol, 'Precious Metals' as sector, 
            price_per_gram as price, updated_at as date
            FROM gold_prices 
            WHERE updated_at >= %s
            ORDER BY updated_at
        """
        return pd.read_sql(query, conn, params=(start_date,))
    except Exception as e:
        print(f"Error fetching gold: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def process_asset_sector_scores(df, asset_type):
    """Calculate sector scores for stocks or mutual funds"""
    if df.empty:
        return pd.DataFrame()
        
    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    sectors = df["sector"].unique()

    output = []
    for sector in sectors:
        sector_df = df[df["sector"] == sector].copy()
        if sector_df.empty:
            continue

        grouped = sector_df.groupby("symbol")
        sector_scores = []
        
        for symbol, group in grouped:
            group = group.sort_values("date")
            if len(group) < 2:
                # For single data point, use change_percent if available
                if "change_percent" in group.columns and not pd.isna(group.iloc[0]["change_percent"]):
                    returns = group.iloc[0]["change_percent"]
                    volatility = 0  # No volatility for single point
                else:
                    continue
            else:
                # Calculate returns from price data
                start_price = group.iloc[0]["close"]
                end_price = group.iloc[-1]["close"]
                returns = ((end_price - start_price) / start_price) * 100 if start_price != 0 else 0
                volatility = group["close"].pct_change().std() * 100
                
            score = returns - (volatility * 0.5)  # Weight volatility less

            sector_scores.append({
                "symbol": symbol,
                "return_%": round(returns, 2),
                "volatility_%": round(volatility, 2) if not pd.isna(volatility) else 0,
                "score": round(score, 2)
            })

        if not sector_scores:
            continue

        avg_return = sum(x["return_%"] for x in sector_scores) / len(sector_scores)
        avg_volatility = sum(x["volatility_%"] for x in sector_scores) / len(sector_scores)
        avg_score = sum(x["score"] for x in sector_scores) / len(sector_scores)

        output.append({
            "sector": sector,
            "asset_type": asset_type,
            "avg_return_%": round(avg_return, 2),
            "avg_volatility_%": round(avg_volatility, 2),
            "avg_score": round(avg_score, 2),
            "investments_analyzed": len(sector_scores),
            "top_performer": max(sector_scores, key=lambda x: x["score"])["symbol"] if sector_scores else "N/A"
        })

    return pd.DataFrame(output).sort_values("avg_score", ascending=False)

def process_gold_score(df):
    """Calculate gold investment score"""
    if df.empty:
        return pd.DataFrame()
        
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if len(df) < 2:
        return pd.DataFrame([{
            "sector": "Precious Metals",
            "asset_type": "gold",
            "avg_return_%": 0.0,
            "avg_volatility_%": 0.0,
            "avg_score": 0.0,
            "investments_analyzed": 1,
            "top_performer": "Gold"
        }])

    start_price = df.iloc[0]["price"]
    end_price = df.iloc[-1]["price"]
    returns = ((end_price - start_price) / start_price) * 100 if start_price != 0 else 0
    volatility = df["price"].pct_change().std() * 100
    score = returns - (volatility * 0.5)

    return pd.DataFrame([{
        "sector": "Precious Metals",
        "asset_type": "gold",
        "avg_return_%": round(returns, 2),
        "avg_volatility_%": round(volatility, 2) if not pd.isna(volatility) else 0,
        "avg_score": round(score, 2),
        "investments_analyzed": 1,
        "top_performer": "Gold"
    }])

def calculate_comprehensive_sector_performance(df):
    """Calculate sector performance across all investment types (stocks, mutual funds, gold)"""
    if df.empty:
        print("No data available for analysis")
        return pd.DataFrame()
    
    print(f"Analyzing {len(df)} investment records across sectors...")
    
    # Convert last_updated to datetime
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    
    # Group by sector and calculate metrics
    sector_results = []
    
    for sector, group in df.groupby('sector'):
        if len(group) < 1:  # Skip empty sectors
            continue
            
        avg_return = group['change_percent'].mean()
        avg_price = group['last_price'].mean()
        investment_count = len(group)
        volatility = group['change_percent'].std()
        
        # Count investment types in this sector
        investment_types = []
        if any('.' in str(symbol) or len(str(symbol)) <= 10 for symbol in group['symbol']):
            investment_types.append('Stocks')
        if any('fund' in str(name).lower() or 'scheme' in str(name).lower() for name in group['name']):
            investment_types.append('Mutual Funds')
        if 'Gold' in sector or 'Precious' in sector:
            investment_types.append('Gold')
        
        # Get top performing investments in the sector
        top_investments = group.nlargest(3, 'change_percent')[['symbol', 'name', 'change_percent']].to_dict('records')
        
        sector_results.append({
            'sector': sector,
            'investment_count': investment_count,
            'investment_types': ', '.join(investment_types) if investment_types else 'Mixed',
            'avg_return_pct': round(avg_return, 2),
            'avg_price': round(avg_price, 2),
            'volatility': round(volatility, 2) if volatility and not pd.isna(volatility) else 0,
            'top_performers': top_investments
        })
    
    sector_df = pd.DataFrame(sector_results)
    
    if not sector_df.empty:
        # Create a comprehensive momentum score
        sector_df['momentum_score'] = (
            sector_df['avg_return_pct'] * 0.6 - 
            sector_df['volatility'] * 0.2 +
            sector_df['investment_count'] * 0.2  # Bonus for diversified sectors
        )
        
        # Sort by momentum score
        sector_df = sector_df.sort_values(by='momentum_score', ascending=False)
    
    return sector_df

def print_comprehensive_sector_analysis(sector_df):
    """Print a formatted comprehensive sector analysis report"""
    if sector_df.empty:
        print("❌ No sector data available for analysis")
        return
    
    print("🏢 COMPREHENSIVE INVESTMENT SECTOR ANALYSIS")
    print("=" * 90)
    print(f"📊 Analyzed {len(sector_df)} sectors across Stocks, Mutual Funds, and Gold")
    print()
    
    for idx, row in sector_df.iterrows():
        print(f"🏆 #{idx+1} {row['sector']}")
        print(f"   💼 Investment Types: {row['investment_types']}")
        print(f"   📈 Avg Return: {row['avg_return_pct']}%")
        print(f"   💰 Avg Price: ₹{row['avg_price']:,.2f}")
        print(f"   📊 Investments: {row['investment_count']}")
        print(f"   ⚡ Volatility: {row['volatility']}%")
        print(f"   🎯 Momentum Score: {row['momentum_score']:.2f}")
        
        if row['top_performers']:
            print("   🌟 Top Performers:")
            for investment in row['top_performers']:
                print(f"      • {investment['symbol']} ({investment['name'][:40]}...): {investment['change_percent']}%")
        print()

# Keep the original function for backward compatibility
def calculate_sector_performance(df):
    """Calculate sector performance based on current stock data (legacy function)"""
    return calculate_comprehensive_sector_performance(df)

def print_sector_analysis(sector_df):
    """Print sector analysis report (legacy function)"""
    return print_comprehensive_sector_analysis(sector_df)
    """Print formatted sector scores"""
    print("\n🏆 ENHANCED SECTOR SCORING ANALYSIS")
    print("=" * 80)
    
    for asset_type, scores in scores_dict.items():
        if not scores:
            continue
            
        print(f"\n📊 {asset_type.upper()} SECTOR SCORES")
        print("-" * 50)
        
        df = pd.DataFrame(scores)
        if df.empty:
            print("   No data available")
            continue
            
        for _, row in df.iterrows():
            print(f"🏢 {row['sector']}")
            print(f"   📈 Avg Return: {row['avg_return_%']}%")
            print(f"   ⚡ Avg Volatility: {row['avg_volatility_%']}%") 
            print(f"   🎯 Score: {row['avg_score']}")
            print(f"   📊 Investments: {row['investments_analyzed']}")
            print(f"   🌟 Top Performer: {row['top_performer']}")
            print()

def fetch_all_investment_data():
    """Fetch combined data from stocks, mutual funds, and gold"""
    print("📊 Fetching investment data from all sources...")
    
    # Fetch data from all sources
    stock_df = fetch_data()
    mf_df = fetch_mutual_fund_data()
    gold_df = fetch_gold_data()
    
    print(f"   📈 Stocks: {len(stock_df)} records")
    print(f"   💰 Mutual Funds: {len(mf_df)} records") 
    print(f"   🏆 Gold: {len(gold_df)} records")
    
    # Combine all dataframes
    if not stock_df.empty or not mf_df.empty or not gold_df.empty:
        combined_df = pd.concat([stock_df, mf_df, gold_df], ignore_index=True)
        print(f"   🔗 Combined: {len(combined_df)} total records")
        return combined_df
    else:
        print("   ❌ No data available from any source")
        return pd.DataFrame()

def fetch_data():
    """Fetch stock data from the database for sector analysis"""
    conn = get_postgres_connection()
    if not conn:
        print("Failed to connect to database")
        return pd.DataFrame()
    
    try:
        # Query the actual stocks table with available columns
        query = """
            SELECT symbol, name, sector, last_price, change_percent, last_updated
            FROM stocks
            WHERE last_updated >= %s
            ORDER BY last_updated ASC
        """
        start_date = datetime.now() - timedelta(days=30)  # Use 30 days for recent data
        df = pd.read_sql(query, conn, params=(start_date,))
        return df
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def fetch_mutual_fund_data():
    """Fetch mutual fund data from the database"""
    conn = get_postgres_connection()
    if not conn:
        print("Failed to connect to database")
        return pd.DataFrame()
    
    try:
        # Query the mutual_funds table
        query = """
            SELECT fund_name AS symbol, fund_name AS name, category AS sector, 
            nav AS last_price, returns_1y AS change_percent, last_updated
            FROM mutual_funds
            WHERE last_updated >= %s AND nav > 0
            ORDER BY last_updated ASC
        """
        start_date = datetime.now() - timedelta(days=30)
        df = pd.read_sql(query, conn, params=(start_date,))
        return df
    except Exception as e:
        print(f"Error fetching mutual fund data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def fetch_gold_data():
    """Fetch gold data from the database"""
    conn = get_postgres_connection()
    if not conn:
        print("Failed to connect to database")
        return pd.DataFrame()
    
    try:
        # Query the gold_prices table
        query = """
            SELECT 'Gold' AS symbol, 'Gold Investment' AS name, 'Precious Metals' AS sector,
            price_per_gram AS last_price, 0 AS change_percent, updated_at AS last_updated
            FROM gold_prices
            WHERE updated_at >= %s
            ORDER BY updated_at ASC
        """
        start_date = datetime.now() - timedelta(days=30)
        df = pd.read_sql(query, conn, params=(start_date,))
        return df
    except Exception as e:
        print(f"Error fetching gold data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
    """Calculate sector performance across all investment types (stocks, mutual funds, gold)"""
    if df.empty:
        print("No data available for analysis")
        return pd.DataFrame()
    
    print(f"Analyzing {len(df)} investment records across sectors...")
    
    # Convert last_updated to datetime
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    
    # Group by sector and calculate metrics
    sector_results = []
    
    for sector, group in df.groupby('sector'):
        if len(group) < 1:  # Skip empty sectors
            continue
            
        avg_return = group['change_percent'].mean()
        avg_price = group['last_price'].mean()
        investment_count = len(group)
        volatility = group['change_percent'].std()
        
        # Count investment types in this sector
        investment_types = []
        if any('.' in str(symbol) or len(str(symbol)) <= 10 for symbol in group['symbol']):
            investment_types.append('Stocks')
        if any('fund' in str(name).lower() or 'scheme' in str(name).lower() for name in group['name']):
            investment_types.append('Mutual Funds')
        if 'Gold' in sector or 'Precious' in sector:
            investment_types.append('Gold')
        
        # Get top performing investments in the sector
        top_investments = group.nlargest(3, 'change_percent')[['symbol', 'name', 'change_percent']].to_dict('records')
        
        sector_results.append({
            'sector': sector,
            'investment_count': investment_count,
            'investment_types': ', '.join(investment_types) if investment_types else 'Mixed',
            'avg_return_pct': round(avg_return, 2),
            'avg_price': round(avg_price, 2),
            'volatility': round(volatility, 2) if volatility and not pd.isna(volatility) else 0,
            'top_performers': top_investments
        })
    
    sector_df = pd.DataFrame(sector_results)
    
    if not sector_df.empty:
        # Create a comprehensive momentum score
        sector_df['momentum_score'] = (
            sector_df['avg_return_pct'] * 0.6 - 
            sector_df['volatility'] * 0.2 +
            sector_df['investment_count'] * 0.2  # Bonus for diversified sectors
        )
        
        # Sort by momentum score
        sector_df = sector_df.sort_values(by='momentum_score', ascending=False)
    
    return sector_df


def print_comprehensive_sector_analysis(sector_df):
    """Print a formatted comprehensive sector analysis report"""
    if sector_df.empty:
        print("❌ No sector data available for analysis")
        return
    
    print("🏢 COMPREHENSIVE INVESTMENT SECTOR ANALYSIS")
    print("=" * 90)
    print(f"📊 Analyzed {len(sector_df)} sectors across Stocks, Mutual Funds, and Gold")
    print()
    
    for idx, row in sector_df.iterrows():
        print(f"🏆 #{idx+1} {row['sector']}")
        print(f"   💼 Investment Types: {row['investment_types']}")
        print(f"   📈 Avg Return: {row['avg_return_pct']}%")
        print(f"   💰 Avg Price: ₹{row['avg_price']:,.2f}")
        print(f"   📊 Investments: {row['investment_count']}")
        print(f"   ⚡ Volatility: {row['volatility']}%")
        print(f"   🎯 Momentum Score: {row['momentum_score']:.2f}")
        
        if row['top_performers']:
            print("   🌟 Top Performers:")
            for investment in row['top_performers']:
                print(f"      • {investment['symbol']} ({investment['name'][:40]}...): {investment['change_percent']}%")
        print()

# Keep the original function for backward compatibility
def calculate_sector_performance(df):
    """Calculate sector performance based on current stock data (legacy function)"""
    return calculate_comprehensive_sector_performance(df)

def print_sector_analysis(sector_df):
    """Print sector analysis report (legacy function)"""
    return print_comprehensive_sector_analysis(sector_df)

def print_sector_scores(scores_dict):
    """Print formatted sector scores"""
    print("\n🏆 ENHANCED SECTOR SCORING ANALYSIS")
    print("=" * 80)
    
    for asset_type, scores in scores_dict.items():
        if not scores:
            continue
            
        print(f"\n📊 {asset_type.upper()} SECTOR SCORES")
        print("-" * 50)
        
        df = pd.DataFrame(scores)
        if df.empty:
            print("   No data available")
            continue
            
        for _, row in df.iterrows():
            print(f"🏢 {row['sector']}")
            print(f"   📈 Avg Return: {row['avg_return_%']}%")
            print(f"   ⚡ Avg Volatility: {row['avg_volatility_%']}%") 
            print(f"   🎯 Score: {row['avg_score']}")
            print(f"   📊 Investments: {row['investments_analyzed']}")
            print(f"   🌟 Top Performer: {row['top_performer']}")
            print()

if __name__ == "__main__":
    print("🔍 Starting Advanced Investment Sector Analysis...")
    print("=" * 70)
    
    # Option 1: Enhanced Sector Scoring Analysis
    print("🌟 Option 1: Enhanced Sector Scoring Analysis")
    print("-" * 50)
    try:
        scores_result = calculate_sector_scores()
        print_sector_scores(scores_result)
        
        # Save enhanced scoring results
        all_scores = []
        for asset_type, scores in scores_result.items():
            all_scores.extend(scores)
        
        if all_scores:
            scores_df = pd.DataFrame(all_scores)
            output_file = f"enhanced_sector_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            scores_df.to_csv(output_file, index=False)
            print(f"💾 Enhanced scoring results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Enhanced scoring failed: {e}")
    
    print("\n" + "="*70)
    print("🌟 Option 2: Comprehensive Analysis (All Investment Types)")
    print("-" * 50)
    
    # Option 2: Comprehensive analysis (original enhanced version)
    combined_df = fetch_all_investment_data()
    
    if not combined_df.empty:
        print(f"✅ Fetched {len(combined_df)} total investment records")
        
        # Calculate comprehensive sector performance  
        comprehensive_summary = calculate_comprehensive_sector_performance(combined_df)
        
        # Print comprehensive results
        print_comprehensive_sector_analysis(comprehensive_summary)
        
        # Save comprehensive results to CSV
        if not comprehensive_summary.empty:
            output_file = f"comprehensive_sector_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            comprehensive_summary.to_csv(output_file, index=False)
            print(f"💾 Comprehensive results saved to: {output_file}")
    else:
        print("❌ No comprehensive data available")
    
    print("\n" + "="*70)
    print("📈 Option 3: Stocks-Only Analysis (Legacy)")
    print("-" * 50)
    
    # Option 3: Stocks-only analysis (original functionality)
    stock_df = fetch_data()
    
    if not stock_df.empty:
        print(f"✅ Fetched {len(stock_df)} stock records")
        
        # Calculate stock sector performance  
        stock_summary = calculate_sector_performance(stock_df)
        
        # Print stock results
        print_sector_analysis(stock_summary)
        
        # Save stock results to CSV
        if not stock_summary.empty:
            output_file = f"stock_sector_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            stock_summary.to_csv(output_file, index=False)
            print(f"💾 Stock-only results saved to: {output_file}")
    else:
        print("❌ No stock data available. Make sure to run the stock scraper first!")
        print("💡 Try running: python -m app.scrapers.stock_scraper")