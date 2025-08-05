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
        """Load stock features directly from stocks table (primary) and sector analysis (supplementary)"""
        try:
            with next(get_db()) as db:
                # First try to load from the main stocks table
                conn = get_postgres_connection()
                if not conn:
                    print("No database connection available")
                    return pd.DataFrame()
                
                # Load stock data with sector information
                stock_query = """
                    SELECT 
                        s.symbol,
                        s.name,
                        s.last_price as current_price,
                        s.change_percent,
                        s.market_cap,
                        s.sector,
                        s.last_updated,
                        (s.change_percent / 100.0) as predicted_return
                    FROM stocks s
                    WHERE s.last_updated >= CURRENT_DATE - INTERVAL '7 days'
                    ORDER BY s.change_percent DESC
                """
                
                stock_df = pd.read_sql(stock_query, conn)
                
                if stock_df.empty:
                    print("No recent stock data found, trying sector analysis...")
                    # Fallback to sector analysis data expansion
                    records = db.query(StockSectorAnalysis).all()
                    
                    if not records:
                        print("No stock sector analysis data found in database")
                        conn.close()
                        return pd.DataFrame()
                    
                    # Convert sector analysis to individual stock records
                    stock_data = []
                    for record in records:
                        if record.investment_types == 'Stocks' and record.top_performers:
                            try:
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
                    
                    stock_df = pd.DataFrame(stock_data)
                else:
                    # Add supplementary data from sector analysis
                    try:
                        sector_query = """
                            SELECT sector, volatility, momentum_score, avg_return_pct
                            FROM stock_sector_analysis
                            WHERE investment_types = 'Stocks'
                        """
                        sector_df = pd.read_sql(sector_query, conn)
                        
                        # Merge with sector data for additional metrics
                        stock_df = stock_df.merge(
                            sector_df, 
                            on='sector', 
                            how='left'
                        )
                        
                        # Fill missing volatility with default values
                        stock_df['volatility'] = stock_df['volatility'].fillna(15.0)
                        stock_df['momentum_score'] = stock_df['momentum_score'].fillna(0.0)
                        stock_df['avg_return_pct'] = stock_df['avg_return_pct'].fillna(stock_df['change_percent'])
                        
                    except Exception as e:
                        print(f"Warning: Could not load sector supplementary data: {e}")
                        # Add default values for missing columns
                        stock_df['volatility'] = 15.0
                        stock_df['momentum_score'] = 0.0
                        stock_df['avg_return_pct'] = stock_df['change_percent']
                
                conn.close()
                print(f"✅ Loaded {len(stock_df)} stocks from database (stocks table)")
                return stock_df
                
        except Exception as e:
            print(f"Error loading stock features from database: {e}")
            return pd.DataFrame()
    
    def load_mutual_fund_features_from_db(self) -> pd.DataFrame:
        """Load mutual fund features directly from mutual_funds table"""
        try:
            conn = get_postgres_connection()
            if not conn:
                return pd.DataFrame()
            
            query = """
                SELECT 
                    fund_name as name,
                    fund_house as fund_manager,
                    category as fund_type,
                    nav as current_nav,
                    returns_1y as return_1year,
                    returns_3y as return_3year,
                    risk_level,
                    last_updated,
                    category
                FROM mutual_funds
                WHERE last_updated >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY returns_1y DESC NULLS LAST
                LIMIT 500
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            # Add calculated fields
            if not df.empty:
                df['predicted_return'] = df['return_1year'].fillna(12.0)  # Default 12% if missing
                df['expected_return'] = df['predicted_return']
                df['current_performance'] = df['return_1year'].fillna(0)
                
                # Map risk levels to numeric scores
                df['risk_score'] = df['risk_level'].map({
                    'Low': 1, 'Medium': 2, 'Moderate': 2, 'High': 3
                }).fillna(2)
                
                # Add investment recommendations
                df['is_sip_recommended'] = True  # Most mutual funds are SIP suitable
                df['minimum_investment'] = 500   # Standard minimum
                df['expense_ratio'] = 1.5        # Default expense ratio
                
                print(f"✅ Loaded {len(df)} mutual funds from database")
            
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
