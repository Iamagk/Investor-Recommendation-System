"""
Database-based sector analysis - stores results directly in database tables
Replaces CSV file generation with direct database operations
"""

import sys
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text, delete
from app.db import get_postgres_connection, get_db, engine
from app.models.analysis_models import (
    StockSectorAnalysis, 
    ComprehensiveSectorAnalysis, 
    EnhancedSectorScores
)

# Number of days to consider for return calculation
LOOKBACK_DAYS = 30

class DatabaseSectorAnalyzer:
    """
    Sector analysis that stores results directly in database tables
    """
    
    def __init__(self):
        self.timestamp = datetime.now()
    
    def log(self, message: str):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def fetch_stock_data(self) -> pd.DataFrame:
        """Fetch stock data from the database"""
        conn = get_postgres_connection()
        if not conn:
            self.log("Failed to connect to database")
            return pd.DataFrame()
        
        try:
            query = """
                SELECT symbol, name, sector, last_price, change_percent, last_updated
                FROM stocks
                WHERE last_updated >= %s
                ORDER BY last_updated ASC
            """
            start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
            df = pd.read_sql(query, conn, params=(start_date,))
            return df
        except Exception as e:
            self.log(f"Error fetching stock data: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def fetch_mutual_fund_data(self) -> pd.DataFrame:
        """Fetch mutual fund data from the database"""
        conn = get_postgres_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
                SELECT fund_name as name, category as sector, nav as last_price, 
                       returns_1y as change_percent, last_updated
                FROM mutual_funds
                WHERE last_updated >= %s
                ORDER BY last_updated ASC
            """
            start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
            df = pd.read_sql(query, conn, params=(start_date,))
            return df
        except Exception as e:
            self.log(f"Error fetching mutual fund data: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def fetch_gold_data(self) -> pd.DataFrame:
        """Fetch gold data from the database"""
        conn = get_postgres_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
                SELECT rate_type as name, 'Gold' as sector, rate as last_price,
                       0 as change_percent, last_updated
                FROM gold_rates
                WHERE last_updated >= %s
                ORDER BY last_updated ASC
            """
            start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
            df = pd.read_sql(query, conn, params=(start_date,))
            return df
        except Exception as e:
            self.log(f"Error fetching gold data: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def calculate_sector_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate sector-wise performance metrics"""
        if df.empty:
            return {}
        
        # Group by sector and calculate metrics
        sector_stats = df.groupby('sector').agg({
            'symbol': 'count',  # Investment count
            'last_price': 'mean',  # Average price
            'change_percent': ['mean', 'std'],  # Return and volatility
        }).round(2)
        
        # Flatten column names
        sector_stats.columns = ['investment_count', 'avg_price', 'avg_return_pct', 'volatility']
        sector_stats['volatility'] = sector_stats['volatility'].fillna(0)
        
        # Calculate momentum score (simple: avg_return * 0.7 + volatility_penalty)
        sector_stats['momentum_score'] = (
            sector_stats['avg_return_pct'] * 0.7 - sector_stats['volatility'] * 0.3
        ).round(3)
        
        # Get top performers for each sector
        sector_performance = {}
        for sector in df['sector'].unique():
            sector_df = df[df['sector'] == sector]
            top_performers = sector_df.nlargest(3, 'change_percent')[
                ['symbol', 'name', 'change_percent']
            ].to_dict('records')
            
            sector_performance[sector] = {
                'investment_count': int(sector_stats.loc[sector, 'investment_count']),
                'investment_types': 'Stocks',  # Default for stocks
                'avg_return_pct': float(sector_stats.loc[sector, 'avg_return_pct']),
                'avg_price': float(sector_stats.loc[sector, 'avg_price']),
                'volatility': float(sector_stats.loc[sector, 'volatility']),
                'top_performers': json.dumps(top_performers),
                'momentum_score': float(sector_stats.loc[sector, 'momentum_score'])
            }
        
        return sector_performance
    
    def store_stock_sector_analysis(self, analysis_data: Dict[str, Any]):
        """Store stock sector analysis in database"""
        try:
            with next(get_db()) as db:
                # Clear existing data
                db.execute(delete(StockSectorAnalysis))
                
                # Insert new data
                for sector, data in analysis_data.items():
                    record = StockSectorAnalysis(
                        sector=sector,
                        investment_count=data['investment_count'],
                        investment_types=data['investment_types'],
                        avg_return_pct=data['avg_return_pct'],
                        avg_price=data['avg_price'],
                        volatility=data['volatility'],
                        top_performers=data['top_performers'],
                        momentum_score=data['momentum_score']
                    )
                    db.add(record)
                
                db.commit()
                self.log(f"✅ Stored {len(analysis_data)} stock sector analysis records")
        
        except Exception as e:
            self.log(f"❌ Error storing stock sector analysis: {e}")
    
    def store_comprehensive_analysis(self, analysis_data: Dict[str, Any]):
        """Store comprehensive sector analysis in database"""
        try:
            with next(get_db()) as db:
                # Clear existing data
                db.execute(delete(ComprehensiveSectorAnalysis))
                
                # Insert new data
                for sector, data in analysis_data.items():
                    record = ComprehensiveSectorAnalysis(
                        sector=sector,
                        investment_count=data['investment_count'],
                        investment_types=data['investment_types'],
                        avg_return_pct=data['avg_return_pct'],
                        avg_price=data['avg_price'],
                        volatility=data['volatility'],
                        top_performers=data['top_performers'],
                        momentum_score=data['momentum_score']
                    )
                    db.add(record)
                
                db.commit()
                self.log(f"✅ Stored {len(analysis_data)} comprehensive analysis records")
        
        except Exception as e:
            self.log(f"❌ Error storing comprehensive analysis: {e}")
    
    def store_enhanced_scores(self, scores_data: List[Dict[str, Any]]):
        """Store enhanced sector scores in database"""
        try:
            with next(get_db()) as db:
                # Clear existing data
                db.execute(delete(EnhancedSectorScores))
                
                # Insert new data
                for score in scores_data:
                    record = EnhancedSectorScores(
                        sector=score['sector'],
                        asset_type=score['asset_type'],
                        avg_return_percent=score['avg_return_%'],
                        avg_volatility_percent=score['avg_volatility_%'],
                        avg_score=score['avg_score'],
                        investments_analyzed=score['investments_analyzed'],
                        top_performer=score['top_performer']
                    )
                    db.add(record)
                
                db.commit()
                self.log(f"✅ Stored {len(scores_data)} enhanced sector scores")
        
        except Exception as e:
            self.log(f"❌ Error storing enhanced scores: {e}")
    
    def run_full_analysis(self):
        """Run complete sector analysis and store in database"""
        self.log("🚀 Starting database-based sector analysis...")
        
        # 1. Fetch data from database
        self.log("📊 Fetching stock data...")
        stock_df = self.fetch_stock_data()
        
        self.log("💰 Fetching mutual fund data...")
        mf_df = self.fetch_mutual_fund_data()
        
        self.log("🏆 Fetching gold data...")
        gold_df = self.fetch_gold_data()
        
        # 2. Calculate and store stock sector analysis
        if not stock_df.empty:
            self.log("🔍 Analyzing stock sectors...")
            stock_analysis = self.calculate_sector_performance(stock_df)
            self.store_stock_sector_analysis(stock_analysis)
        
        # 3. Calculate comprehensive analysis (all investment types combined)
        if not stock_df.empty or not mf_df.empty or not gold_df.empty:
            self.log("🔗 Creating comprehensive analysis...")
            
            # Combine all data sources
            all_dfs = []
            if not stock_df.empty:
                stock_df['investment_type'] = 'Stocks'
                all_dfs.append(stock_df)
            if not mf_df.empty:
                mf_df['investment_type'] = 'Mutual Funds'
                all_dfs.append(mf_df)
            if not gold_df.empty:
                gold_df['investment_type'] = 'Gold'
                all_dfs.append(gold_df)
            
            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                comprehensive_analysis = self.calculate_sector_performance(combined_df)
                self.store_comprehensive_analysis(comprehensive_analysis)
        
        # 4. Generate enhanced scores (simplified version)
        if not stock_df.empty:
            self.log("⚡ Generating enhanced scores...")
            enhanced_scores = []
            
            for sector in stock_df['sector'].unique():
                sector_data = stock_df[stock_df['sector'] == sector]
                enhanced_scores.append({
                    'sector': sector,
                    'asset_type': 'stock',
                    'avg_return_%': float(sector_data['change_percent'].mean()),
                    'avg_volatility_%': float(sector_data['change_percent'].std() or 0),
                    'avg_score': float(sector_data['change_percent'].mean()),
                    'investments_analyzed': len(sector_data),
                    'top_performer': sector_data.loc[sector_data['change_percent'].idxmax(), 'symbol']
                })
            
            self.store_enhanced_scores(enhanced_scores)
        
        self.log("✅ Database-based analysis completed successfully!")

# Standalone functions for backwards compatibility
def run_database_analysis():
    """Run the complete database-based analysis"""
    analyzer = DatabaseSectorAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    run_database_analysis()
