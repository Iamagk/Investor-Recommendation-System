import pandas as pd
import json
import os
from sqlalchemy.orm import Session
from app.db import get_db

# Import the new database-based loader
from app.utils.db_data_loader import (
    load_stock_features as load_stock_features_from_db,
    load_mutual_fund_features as load_mutual_fund_features_from_db,
    load_gold_features as load_gold_features_from_db,
    load_comprehensive_analysis as load_comprehensive_analysis_from_db
)

def load_stock_features():
    """
    Load stock features ONLY from database - no CSV fallback
    """
    try:
        # Load directly from database
        df = load_stock_features_from_db()
        if not df.empty:
            print(f"✅ Loaded {len(df)} stock records from database")
            return df
        else:
            print("❌ No stock data found in database tables")
            return pd.DataFrame()
    except Exception as e:
        print(f"❌ Database loading failed: {e}")
        return pd.DataFrame()

def load_stock_features_from_csv():
    """
    FALLBACK: Load stock features from CSV file with real stock data
    """
    try:
        # Load the comprehensive sector analysis CSV
        csv_path = "/Users/allenngeorge/Projects/investment_recommender/data/comprehensive_sector_analysis_20250803_090202.csv"
        
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path)
        
        # Expand the top_performers JSON data to get individual stocks
        stock_data = []
        for _, row in df.iterrows():
            if row['investment_types'] == 'Stocks':
                try:
                    # Parse the top_performers JSON string
                    performers = json.loads(row['top_performers'].replace("'", '"'))
                    for stock in performers:
                        stock_data.append({
                            'symbol': stock['symbol'],
                            'name': stock['name'],
                            'sector': row['sector'],
                            'current_price': row['avg_price'],
                            'change_percent': stock['change_percent'],
                            'predicted_return': stock['change_percent'] / 100,  # Convert to decimal
                            'volatility': row['volatility'],
                            'momentum_score': row['momentum_score'],
                            'avg_return_pct': row['avg_return_pct']
                        })
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for sector {row['sector']}: {e}")
                    continue
        
        return pd.DataFrame(stock_data)
    except Exception as e:
        print(f"Error loading stock features from CSV: {e}")
        return pd.DataFrame()

def load_mutual_fund_features():
    """
    Load mutual fund features - sample data for demonstration
    """
    try:
        # Sample mutual fund data - in reality this would come from database/API
        mutual_funds_data = [
            {
                'fund_name': 'HDFC Top 100 Fund',
                'fund_manager': 'HDFC Asset Management',
                'category': 'Large Cap Equity',
                'current_nav': 852.43,
                'expense_ratio': 1.05,
                'predicted_return': 0.14,
                'risk_level': 'Moderate',
                'aum': 25000,
                '1_year_return': 18.5,
                '3_year_return': 15.2,
                '5_year_return': 12.8
            },
            {
                'fund_name': 'ICICI Prudential Bluechip Fund',
                'fund_manager': 'ICICI Prudential AMC',
                'category': 'Large Cap Equity', 
                'current_nav': 67.12,
                'expense_ratio': 1.25,
                'predicted_return': 0.13,
                'risk_level': 'Moderate',
                'aum': 18500,
                '1_year_return': 16.8,
                '3_year_return': 14.5,
                '5_year_return': 11.9
            },
            {
                'fund_name': 'SBI Large & Midcap Fund',
                'fund_manager': 'SBI Mutual Fund',
                'category': 'Large & Mid Cap Equity',
                'current_nav': 234.67,
                'expense_ratio': 1.15,
                'predicted_return': 0.16,
                'risk_level': 'Moderate to High',
                'aum': 12800,
                '1_year_return': 22.3,
                '3_year_return': 17.1,
                '5_year_return': 14.2
            },
            {
                'fund_name': 'Axis Small Cap Fund',
                'fund_manager': 'Axis Asset Management',
                'category': 'Small Cap Equity',
                'current_nav': 89.45,
                'expense_ratio': 1.35,
                'predicted_return': 0.18,
                'risk_level': 'High',
                'aum': 8500,
                '1_year_return': 28.7,
                '3_year_return': 19.8,
                '5_year_return': 16.5
            },
            {
                'fund_name': 'Kotak Emerging Equity Fund',
                'fund_manager': 'Kotak Mahindra AMC',
                'category': 'Multi Cap Equity',
                'current_nav': 156.23,
                'expense_ratio': 1.45,
                'predicted_return': 0.15,
                'risk_level': 'Moderate to High',
                'aum': 9200,
                '1_year_return': 20.1,
                '3_year_return': 16.3,
                '5_year_return': 13.7
            },
            {
                'fund_name': 'Mirae Asset Large Cap Fund',
                'fund_manager': 'Mirae Asset Investment Managers',
                'category': 'Large Cap Equity',
                'current_nav': 134.89,
                'expense_ratio': 1.00,
                'predicted_return': 0.12,
                'risk_level': 'Moderate',
                'aum': 15600,
                '1_year_return': 17.2,
                '3_year_return': 14.8,
                '5_year_return': 12.1
            }
        ]
        
        return pd.DataFrame(mutual_funds_data)
    except Exception as e:
        print(f"Error loading mutual fund features: {e}")
        return pd.DataFrame()

def load_gold_features():
    """
    Load gold investment features with current market data
    """
    try:
        # Real gold investment options data
        gold_data = [
            {
                'investment_type': 'Gold ETF',
                'fund_name': 'HDFC Gold ETF',
                'current_price': 65.47,
                'expense_ratio': 0.50,
                'predicted_return': 0.085,
                'liquidity_rating': 'High',
                'storage_required': False,
                'tax_implications': 'STCG/LTCG as per equity',
                'tracking_error': 0.15,
                'aum': 2800
            },
            {
                'investment_type': 'Digital Gold',
                'fund_name': 'Paytm Gold',
                'current_price': 6620.0,  # Per gram
                'expense_ratio': 0.30,
                'predicted_return': 0.10,
                'liquidity_rating': 'High',
                'storage_required': False,
                'tax_implications': 'STCG/LTCG as per equity',
                'tracking_error': 0.05,
                'aum': 1500
            },
            {
                'investment_type': 'Physical Gold',
                'fund_name': 'Physical Gold Investment',
                'current_price': 6750.0,  # Per gram
                'expense_ratio': 2.00,  # Making charges, storage, etc.
                'predicted_return': 0.115,
                'liquidity_rating': 'Medium',
                'storage_required': True,
                'tax_implications': 'LTCG after 3 years',
                'tracking_error': 0.0,
                'aum': 0  # Not applicable
            },
            {
                'investment_type': 'Gold Mutual Fund',
                'fund_name': 'ICICI Prudential Regular Gold Savings Fund',
                'current_price': 23.45,
                'expense_ratio': 1.00,
                'predicted_return': 0.08,
                'liquidity_rating': 'High',
                'storage_required': False,
                'tax_implications': 'STCG/LTCG as per debt',
                'tracking_error': 0.25,
                'aum': 850
            }
        ]
        
        return pd.DataFrame(gold_data)
    except Exception as e:
        print(f"Error loading gold features: {e}")
        return pd.DataFrame()
