# app/utils/data_loader.py

import pandas as pd
from sqlalchemy.orm import Session
from app.db import get_db
from app.models.analysis_models import StockSectorAnalysis
import ast
import json

def load_stock_features():
    """
    Load stock features from database for ML predictions
    Returns a DataFrame with stock features ready for ML processing
    """
    try:
        # Get database session
        db = next(get_db())
        
        # Query stock sector analysis data
        sectors = db.query(StockSectorAnalysis).all()
        
        stock_data = []
        
        for sector in sectors:
            # Parse top performers to get individual stocks
            top_performers = sector.top_performers or ""
            if top_performers:
                try:
                    # Handle different formats of top_performers
                    performers = []
                    if top_performers.startswith('['):
                        try:
                            # Try JSON parsing first
                            performers = json.loads(top_performers.replace("'", '"'))
                        except:
                            try:
                                # Try ast.literal_eval for Python-like strings
                                performers = ast.literal_eval(top_performers)
                            except:
                                # Fallback: split by comma if it's a simple string
                                performers = [{'symbol': s.strip()} for s in top_performers.split(',') if s.strip()]
                    else:
                        # Simple comma-separated symbols
                        performers = [{'symbol': s.strip()} for s in top_performers.split(',') if s.strip()]
                    
                    # Extract features for each stock
                    for performer in performers:
                        if isinstance(performer, dict) and 'symbol' in performer:
                            symbol = performer['symbol']
                            if symbol and len(symbol) <= 20 and not symbol.startswith('['):
                                stock_data.append({
                                    'Symbol': symbol,
                                    'Sector': sector.sector,
                                    'Momentum': sector.momentum_score or 0,
                                    'Volatility': sector.volatility or 0,
                                    'SectorScore': calculate_sector_score(
                                        sector.avg_return_pct or 0,
                                        sector.volatility or 0,
                                        sector.momentum_score or 0
                                    ),
                                    'Close': sector.avg_price or 0,
                                    'AvgReturn': sector.avg_return_pct or 0,
                                    'CompanyName': performer.get('name', symbol),
                                    'ChangePercent': performer.get('change_percent', 0)
                                })
                        elif isinstance(performer, str) and performer.strip():
                            symbol = performer.strip()
                            if len(symbol) <= 20 and not symbol.startswith('['):
                                stock_data.append({
                                    'Symbol': symbol,
                                    'Sector': sector.sector,
                                    'Momentum': sector.momentum_score or 0,
                                    'Volatility': sector.volatility or 0,
                                    'SectorScore': calculate_sector_score(
                                        sector.avg_return_pct or 0,
                                        sector.volatility or 0,
                                        sector.momentum_score or 0
                                    ),
                                    'Close': sector.avg_price or 0,
                                    'AvgReturn': sector.avg_return_pct or 0,
                                    'CompanyName': symbol,
                                    'ChangePercent': 0
                                })
                                
                except Exception as parse_error:
                    print(f"Error parsing top_performers for {sector.sector}: {parse_error}")
                    continue
        
        # Close database session
        db.close()
        
        # Convert to DataFrame
        if stock_data:
            df = pd.DataFrame(stock_data)
            # Remove duplicates based on Symbol
            df = df.drop_duplicates(subset=['Symbol'], keep='first')
            return df
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'Symbol', 'Sector', 'Momentum', 'Volatility', 
                'SectorScore', 'Close', 'AvgReturn', 'CompanyName', 'ChangePercent'
            ])
            
    except Exception as e:
        print(f"Error in load_stock_features: {e}")
        # Return empty DataFrame in case of error
        return pd.DataFrame(columns=[
            'Symbol', 'Sector', 'Momentum', 'Volatility', 
            'SectorScore', 'Close', 'AvgReturn', 'CompanyName', 'ChangePercent'
        ])

def calculate_sector_score(avg_return_pct, volatility, momentum_score):
    """
    Calculate sector score using weighted formula
    """
    WEIGHTS = {
        "return": 0.4,
        "volatility": 0.2,
        "momentum": 0.4,
    }
    
    return (
        WEIGHTS["return"] * avg_return_pct
        - WEIGHTS["volatility"] * volatility
        + WEIGHTS["momentum"] * momentum_score
    )

def load_mutual_fund_features():
    """
    Load mutual fund features from database
    Returns a DataFrame with mutual fund features
    """
    try:
        db = next(get_db())
        
        # This can be expanded to load actual mutual fund data
        # For now, return empty DataFrame
        df = pd.DataFrame(columns=['name', 'nav', 'returns', 'category'])
        
        db.close()
        return df
        
    except Exception as e:
        print(f"Error in load_mutual_fund_features: {e}")
        return pd.DataFrame(columns=['name', 'nav', 'returns', 'category'])

def load_gold_features():
    """
    Load gold price features from database
    Returns a DataFrame with gold price data
    """
    try:
        db = next(get_db())
        
        # This can be expanded to load actual gold price data
        # For now, return empty DataFrame
        df = pd.DataFrame(columns=['date', 'price', 'change', 'volume'])
        
        db.close()
        return df
        
    except Exception as e:
        print(f"Error in load_gold_features: {e}")
        return pd.DataFrame(columns=['date', 'price', 'change', 'volume'])
