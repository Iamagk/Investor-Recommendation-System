"""
Database-based data loader - loads data directly from database tables
Replaces CSV file loading with direct database queries
"""

import pandas as pd
import json
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.db import get_db, get_postgres_connection
from app.models.analysis_models import (
    StockSectorAnalysis, 
    ComprehensiveSectorAnalysis, 
    EnhancedSectorScores
)

class DatabaseDataLoader:
    """Load investment data directly from database tables"""
    
    def __init__(self):
        pass
    
    def load_stock_features_from_db(self) -> pd.DataFrame:
        """Load stock features directly from database"""
        try:
            with next(get_db()) as db:
                # Query stock sector analysis table
                records = db.query(StockSectorAnalysis).all()
                
                if not records:
                    print("No stock sector analysis data found in database")
                    return pd.DataFrame()
                
                # Convert to DataFrame and expand top performers
                stock_data = []
                for record in records:
                    if record.investment_types == 'Stocks' and record.top_performers:
                        try:
                            # Parse the top_performers JSON string
                            performers = json.loads(record.top_performers)
                            for stock in performers:
                                stock_data.append({
                                    'symbol': stock['symbol'],
                                    'name': stock['name'],
                                    'sector': record.sector,
                                    'current_price': record.avg_price,
                                    'change_percent': stock['change_percent'],
                                    'predicted_return': stock['change_percent'] / 100,
                                    'volatility': record.volatility,
                                    'momentum_score': record.momentum_score,
                                    'avg_return_pct': record.avg_return_pct
                                })
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Error parsing top performers for sector {record.sector}: {e}")
                            continue
                
                return pd.DataFrame(stock_data)
                
        except Exception as e:
            print(f"Error loading stock features from database: {e}")
            return pd.DataFrame()
    
    def load_mutual_fund_features_from_db(self) -> pd.DataFrame:
        """Load mutual fund features directly from database"""
        try:
            conn = get_postgres_connection()
            if not conn:
                return pd.DataFrame()
            
            query = """
                SELECT 
                    fund_name as name,
                    category as fund_type,
                    nav as current_nav,
                    returns_1y as return_1year,
                    returns_3y as return_3year,
                    risk_level,
                    'Debt' as category
                FROM mutual_funds
                WHERE last_updated >= CURRENT_DATE - INTERVAL '30 days'
                LIMIT 100
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            # Add calculated fields
            if not df.empty:
                df['predicted_return'] = df['return_1year'].fillna(0) / 100
                df['risk_score'] = df['risk_level'].map({
                    'Low': 1, 'Moderate': 2, 'High': 3
                }).fillna(2)
            
            return df
            
        except Exception as e:
            print(f"Error loading mutual fund features from database: {e}")
            return pd.DataFrame()
    
    def load_gold_features_from_db(self) -> pd.DataFrame:
        """Load gold features directly from database"""
        try:
            conn = get_postgres_connection()
            if not conn:
                return pd.DataFrame()
            
            query = """
                SELECT 
                    rate_type as name,
                    rate as current_price,
                    'Gold' as type,
                    0 as predicted_return,
                    'Physical' as category
                FROM gold_rates
                WHERE last_updated >= CURRENT_DATE - INTERVAL '7 days'
                ORDER BY last_updated DESC
                LIMIT 10
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            print(f"Error loading gold features from database: {e}")
            return pd.DataFrame()
    
    def load_comprehensive_analysis_from_db(self) -> pd.DataFrame:
        """Load comprehensive sector analysis from database"""
        try:
            with next(get_db()) as db:
                records = db.query(ComprehensiveSectorAnalysis).all()
                
                if not records:
                    print("No comprehensive analysis data found in database")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for record in records:
                    data.append({
                        'sector': record.sector,
                        'investment_count': record.investment_count,
                        'investment_types': record.investment_types,
                        'avg_return_pct': record.avg_return_pct,
                        'avg_price': record.avg_price,
                        'volatility': record.volatility,
                        'top_performers': record.top_performers,
                        'momentum_score': record.momentum_score
                    })
                
                return pd.DataFrame(data)
                
        except Exception as e:
            print(f"Error loading comprehensive analysis from database: {e}")
            return pd.DataFrame()
    
    def load_enhanced_scores_from_db(self) -> pd.DataFrame:
        """Load enhanced sector scores from database"""
        try:
            with next(get_db()) as db:
                records = db.query(EnhancedSectorScores).all()
                
                if not records:
                    print("No enhanced scores data found in database")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for record in records:
                    data.append({
                        'sector': record.sector,
                        'asset_type': record.asset_type,
                        'avg_return_%': record.avg_return_percent,
                        'avg_volatility_%': record.avg_volatility_percent,
                        'avg_score': record.avg_score,
                        'investments_analyzed': record.investments_analyzed,
                        'top_performer': record.top_performer
                    })
                
                return pd.DataFrame(data)
                
        except Exception as e:
            print(f"Error loading enhanced scores from database: {e}")
            return pd.DataFrame()

# Create global instance for backwards compatibility
db_loader = DatabaseDataLoader()

# Backwards compatibility functions
def load_stock_features():
    """Load stock features from database (backwards compatible)"""
    return db_loader.load_stock_features_from_db()

def load_mutual_fund_features():
    """Load mutual fund features from database (backwards compatible)"""
    return db_loader.load_mutual_fund_features_from_db()

def load_gold_features():
    """Load gold features from database (backwards compatible)"""
    return db_loader.load_gold_features_from_db()

def load_comprehensive_analysis():
    """Load comprehensive analysis from database (backwards compatible)"""
    return db_loader.load_comprehensive_analysis_from_db()

def load_enhanced_scores():
    """Load enhanced scores from database (backwards compatible)"""
    return db_loader.load_enhanced_scores_from_db()
