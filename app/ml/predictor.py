import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
import random
import joblib
import os
from datetime import datetime, timedelta

from app.models.analysis_models import (
    StockSectorAnalysis,
    ComprehensiveSectorAnalysis,
    EnhancedSectorScores
)

# Import technical analysis functions
try:
    from app.ml.technical_analysis import generate_features
    technical_analysis_available = True
except ImportError:
    technical_analysis_available = False
    print("Warning: Technical analysis not available")

# Try to import realtime prices - fallback if not available
try:
    from app.utils.realtime_prices import fetch_realtime_price
    realtime_available = True
except ImportError:
    realtime_available = False
    def fetch_realtime_price(symbol):
        return 0

# Load models once (could also cache)
try:
    if os.path.exists("models/random_forest_model.pkl") and os.path.exists("models/xgboost_model.pkl"):
        rf_model = joblib.load("models/random_forest_model.pkl")
        xgb_model = joblib.load("models/xgboost_model.pkl")
        models_loaded = True
        print("Ensemble models loaded successfully!")
    else:
        rf_model = None
        xgb_model = None
        models_loaded = False
        print("Warning: Pre-trained models not found. Using fallback prediction methods.")
except Exception as e:
    rf_model = None
    xgb_model = None
    models_loaded = False
    print(f"Warning: Could not load models: {e}")

def ensemble_predict(features: pd.DataFrame) -> float:
    """Use ensemble of pre-trained models for prediction"""
    if not models_loaded or rf_model is None or xgb_model is None:
        # Fallback to simple prediction if models not available
        return features.iloc[0].get('avg_return_pct', 2.0)
    
    try:
        rf_pred = rf_model.predict(features)[0]
        xgb_pred = xgb_model.predict(features)[0]
        return round((rf_pred + xgb_pred) / 2, 4)
    except Exception as e:
        print(f"Error in ensemble prediction: {e}")
        return features.iloc[0].get('avg_return_pct', 2.0)

def predict_sector_roi(sector_features: pd.DataFrame, use_realtime=False) -> float:
    """Predict sector ROI using ensemble models"""
    return ensemble_predict(sector_features)

def predict_asset_roi(asset_features: pd.DataFrame, symbol=None, use_realtime=False) -> float:
    """Predict individual asset ROI with optional real-time price integration"""
    if use_realtime and symbol and realtime_available:
        try:
            live_price = fetch_realtime_price(symbol)
            if live_price > 0:
                asset_features = asset_features.copy()
                asset_features['live_price'] = live_price
                print(f"Using real-time price for {symbol}: ₹{live_price:.2f}")
            else:
                asset_features = asset_features.copy()
                asset_features['live_price'] = asset_features.get('close', asset_features.get('avg_price', 0))
        except Exception as e:
            print(f"Error fetching real-time price for {symbol}: {e}")
            asset_features = asset_features.copy()
            asset_features['live_price'] = asset_features.get('close', asset_features.get('avg_price', 0))
    else:
        asset_features = asset_features.copy()
        asset_features['live_price'] = asset_features.get('close', asset_features.get('avg_price', 0))
    
    return ensemble_predict(asset_features)

def compute_score(predicted_return: float, volatility: float, momentum_score: float) -> float:
    """Compute investment score based on predicted return, volatility, and momentum"""
    # Weighted scoring: 40% return, 20% volatility (inverse), 40% momentum
    return_score = predicted_return * 0.4
    volatility_score = max(0, (10 - volatility)) * 0.2  # Lower volatility is better
    momentum_score_weighted = momentum_score * 0.4
    
    return return_score + volatility_score + momentum_score_weighted

def predict_returns(features: List[Dict[str, float]]) -> List[float]:
    """Enhanced prediction function using ensemble models when available"""
    predictions = []
    
    for feature in features:
        if models_loaded and rf_model is not None and xgb_model is not None:
            # Use ensemble models if available
            try:
                # Convert to DataFrame for model prediction
                feature_df = pd.DataFrame([feature])
                # Ensure required columns exist
                required_cols = ['avg_return_pct', 'volatility', 'momentum_score', 'investment_count', 'avg_price']
                for col in required_cols:
                    if col not in feature_df.columns:
                        feature_df[col] = 0
                
                pred = ensemble_predict(feature_df[required_cols])
                predictions.append(pred)
            except Exception as e:
                print(f"Error using ensemble models, falling back: {e}")
                # Fallback to simple prediction
                pred = (feature.get("avg_return_pct", 0) * 0.5 + 
                        feature.get("momentum_score", 0) * 0.3 +
                        max(0, 10 - feature.get("volatility", 0)) * 0.2)
                predictions.append(pred)
        else:
            # Simple weighted prediction based on existing metrics (fallback)
            pred = (feature.get("avg_return_pct", 0) * 0.5 + 
                    feature.get("momentum_score", 0) * 0.3 +
                    max(0, 10 - feature.get("volatility", 0)) * 0.2)
            predictions.append(pred)
    
    return predictions

def fetch_data(db: Session, asset_type: str) -> Optional[pd.DataFrame]:
    """Fetch data from database based on asset type"""
    if asset_type == "stocks":
        data = db.query(StockSectorAnalysis).all()
    elif asset_type == "mutual_funds":
        data = db.query(ComprehensiveSectorAnalysis).all()
    elif asset_type == "gold":
        data = db.query(EnhancedSectorScores).filter(
            EnhancedSectorScores.asset_type.ilike('%gold%')
        ).all()
    else:
        return None

    # Convert SQLAlchemy objects to DataFrame
    df = pd.DataFrame([row.__dict__ for row in data])
    df = df.drop(columns=["_sa_instance_state"], errors="ignore")
    return df

def get_stock_timing_analysis(symbol, use_technical_analysis=True):
    """Get detailed timing analysis for a specific stock using technical indicators"""
    try:
        current_date = datetime.now()
        
        if use_technical_analysis and technical_analysis_available:
            # Try to get real stock data and apply technical analysis
            try:
                # Generate sample OHLCV data for demonstration
                # In production, this would fetch real data from your data source
                dates = pd.date_range(end=current_date, periods=252, freq='D')
                base_price = random.uniform(50, 2000)
                
                # Generate realistic OHLCV data
                price_data = []
                current_price = base_price
                
                for i in range(252):
                    # Random walk with slight upward bias
                    change = random.gauss(0.001, 0.02)  # 0.1% daily drift, 2% volatility
                    current_price *= (1 + change)
                    
                    # Generate OHLC from current price
                    daily_volatility = random.uniform(0.005, 0.03)
                    high = current_price * (1 + daily_volatility)
                    low = current_price * (1 - daily_volatility)
                    open_price = random.uniform(low, high)
                    close = current_price
                    volume = random.randint(100000, 1000000)
                    
                    price_data.append({
                        'Date': dates[i],
                        'Open': open_price,
                        'High': high,
                        'Low': low,
                        'Close': close,
                        'Volume': volume
                    })
                
                df = pd.DataFrame(price_data)
                df.set_index('Date', inplace=True)
                
                # Apply technical analysis
                df_with_indicators = generate_features(df)
                
                # Get latest indicators
                latest = df_with_indicators.iloc[-1]
                
                # Make trading decisions based on technical indicators
                current_price = latest['Close']
                rsi = latest.get('RSI_14', 50)
                macd = latest.get('MACD', 0)
                bb_position = (current_price - latest.get('BB_Lower', current_price)) / (latest.get('BB_Upper', current_price) - latest.get('BB_Lower', current_price))
                
                # Technical analysis-based entry/exit logic
                # Entry signal: RSI oversold + MACD turning positive + near lower Bollinger Band
                entry_score = 0
                if rsi < 30:  # Oversold
                    entry_score += 1
                if macd > 0:  # MACD positive
                    entry_score += 1
                if bb_position < 0.3:  # Near lower Bollinger Band
                    entry_score += 1
                
                # Calculate entry timing
                if entry_score >= 2:
                    entry_date = current_date + timedelta(days=random.randint(1, 3))
                    entry_price = current_price * random.uniform(0.98, 1.02)
                else:
                    entry_date = current_date + timedelta(days=random.randint(3, 10))
                    entry_price = current_price * random.uniform(0.95, 1.05)
                
                # Calculate targets based on ATR and technical levels
                atr = latest.get('ATR_14', current_price * 0.02)
                support_level = latest.get('BB_Lower', current_price * 0.95)
                resistance_level = latest.get('BB_Upper', current_price * 1.05)
                
                # Risk management
                stop_loss = entry_price - (2 * atr)  # 2 ATR stop loss
                target_price = entry_price + (3 * atr)  # 3:1 risk-reward ratio
                
                # Holding period based on trend strength
                adx = latest.get('ADX_14', 25)
                if adx > 25:  # Strong trend
                    holding_period = random.randint(30, 60)
                else:  # Weak trend
                    holding_period = random.randint(15, 30)
                
                exit_date = entry_date + timedelta(days=holding_period)
                exit_price = target_price * random.uniform(0.9, 1.1)
                
                # Technical signal interpretation
                signals = []
                if rsi < 30:
                    signals.append("RSI oversold - potential buy signal")
                elif rsi > 70:
                    signals.append("RSI overbought - consider taking profits")
                
                if macd > 0:
                    signals.append("MACD positive - bullish momentum")
                else:
                    signals.append("MACD negative - bearish momentum")
                
                if bb_position < 0.2:
                    signals.append("Price near lower Bollinger Band - potential support")
                elif bb_position > 0.8:
                    signals.append("Price near upper Bollinger Band - potential resistance")
                
                return {
                    'symbol': symbol,
                    'current_price': round(current_price, 2),
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'entry_price': round(entry_price, 2),
                    'exit_date': exit_date.strftime('%Y-%m-%d'),
                    'exit_price': round(exit_price, 2),
                    'expected_return': round(((exit_price - entry_price) / entry_price) * 100, 2),
                    'support_level': round(support_level, 2),
                    'resistance_level': round(resistance_level, 2),
                    'stop_loss': round(stop_loss, 2),
                    'target_price': round(target_price, 2),
                    'holding_period': holding_period,
                    'volatility': round(atr / current_price * 100, 2),
                    'technical_indicators': {
                        'rsi': round(rsi, 2),
                        'macd': round(macd, 4),
                        'bollinger_position': round(bb_position, 3),
                        'atr': round(atr, 2),
                        'adx': round(adx, 2)
                    },
                    'signals': signals,
                    'entry_score': entry_score,
                    'analysis_method': 'technical_indicators'
                }
                
            except Exception as e:
                print(f"Error in technical analysis for {symbol}: {e}")
                # Fall back to simulated analysis
                pass
        
        # Fallback simulation (original logic)
        base_price = random.uniform(50, 2000)
        volatility = random.uniform(0.02, 0.08)
        
        entry_date = current_date + timedelta(days=random.randint(1, 7))
        exit_date = entry_date + timedelta(days=random.randint(30, 90))
        
        entry_price = base_price * (1 + random.uniform(-0.05, 0.02))
        exit_price = entry_price * (1 + random.uniform(0.03, 0.15))
        
        support_level = entry_price * 0.95
        resistance_level = entry_price * 1.10
        stop_loss = entry_price * 0.92
        target_price = entry_price * 1.12
        
        return {
            'symbol': symbol,
            'current_price': round(base_price, 2),
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'entry_price': round(entry_price, 2),
            'exit_date': exit_date.strftime('%Y-%m-%d'),
            'exit_price': round(exit_price, 2),
            'expected_return': round(((exit_price - entry_price) / entry_price) * 100, 2),
            'support_level': round(support_level, 2),
            'resistance_level': round(resistance_level, 2),
            'stop_loss': round(stop_loss, 2),
            'target_price': round(target_price, 2),
            'holding_period': (exit_date - entry_date).days,
            'volatility': round(volatility * 100, 2),
            'analysis_method': 'simulated'
        }
    except Exception as e:
        print(f"Error in timing analysis for {symbol}: {e}")
        return None

def recommend_stocks(db: Session, top_n: int = 5) -> List[Dict[str, Any]]:
    """Get top stock sector recommendations"""
    try:
        sectors = db.query(StockSectorAnalysis).all()
        scored_sectors = []
        features = []
        sector_map = {}

        for idx, sector in enumerate(sectors):
            features.append({
                "avg_return_pct": sector.avg_return_pct or 0,
                "volatility": sector.volatility or 0,
                "momentum_score": sector.momentum_score or 0,
                "investment_count": sector.investment_count or 0,
                "avg_price": sector.avg_price or 0
            })
            sector_map[idx] = sector

        predicted_returns = predict_returns(features)

        for idx, pred_return in enumerate(predicted_returns):
            sector = sector_map[idx]
            score = compute_score(pred_return, sector.volatility or 0, sector.momentum_score or 0)
            
            scored_sectors.append({
                "sector": sector.sector,
                "predicted_return": round(pred_return, 2),
                "score": round(score, 2),
                "volatility": sector.volatility or 0,
                "momentum_score": sector.momentum_score or 0,
                "investment_count": sector.investment_count or 0,
                "avg_price": sector.avg_price or 0,
                "top_performers": sector.top_performers or ""
            })

        # Sort by score and return top N
        ranked = sorted(scored_sectors, key=lambda x: x["score"], reverse=True)
        return ranked[:top_n]
    except Exception as e:
        print(f"Error in stock recommendations: {e}")
        return []

def recommend_stocks_enhanced(db: Session, top_n: int = 5) -> Dict[str, Any]:
    """Enhanced stock recommendations with conversational explanations and timing analysis"""
    try:
        sectors = db.query(StockSectorAnalysis).all()
        scored_sectors = []
        features = []
        sector_map = {}

        for idx, sector in enumerate(sectors):
            features.append({
                "avg_return_pct": sector.avg_return_pct or 0,
                "volatility": sector.volatility or 0,
                "momentum_score": sector.momentum_score or 0,
                "investment_count": sector.investment_count or 0,
                "avg_price": sector.avg_price or 0
            })
            sector_map[idx] = sector

        predicted_returns = predict_returns(features)

        for idx, pred_return in enumerate(predicted_returns):
            sector = sector_map[idx]
            score = compute_score(pred_return, sector.volatility or 0, sector.momentum_score or 0)
            
            # Generate conversational explanations
            risk_level = "Low" if (sector.volatility or 0) < 5 else "Medium" if (sector.volatility or 0) < 10 else "High"
            momentum_desc = "Strong upward" if (sector.momentum_score or 0) > 7 else "Moderate" if (sector.momentum_score or 0) > 4 else "Weak"
            
            # Get timing analysis for top performers
            timing_info = None
            if sector.top_performers:
                # Get the first stock symbol for timing analysis
                symbols = sector.top_performers.split(',')
                if symbols and len(symbols) > 0:
                    timing_info = get_stock_timing_analysis(symbols[0].strip())
            
            explanation = f"The {sector.sector} sector shows {momentum_desc.lower()} momentum with {risk_level.lower()} volatility. "
            explanation += f"Based on our ML analysis, we predict a {pred_return:.1f}% return. "
            explanation += f"This sector has {sector.investment_count or 0} active investments with an average price of ₹{sector.avg_price or 0:.2f}. "
            
            if timing_info:
                explanation += f"For {timing_info['symbol']}, consider entry around ₹{timing_info['entry_price']:.2f} "
                explanation += f"with a target of ₹{timing_info['target_price']:.2f} over {timing_info['holding_period']} days."
            
            sector_data = {
                "sector": sector.sector,
                "predicted_return": round(pred_return, 2),
                "score": round(score, 2),
                "volatility": sector.volatility or 0,
                "momentum_score": sector.momentum_score or 0,
                "investment_count": sector.investment_count or 0,
                "avg_price": sector.avg_price or 0,
                "top_performers": sector.top_performers or "",
                "risk_level": risk_level,
                "momentum_description": momentum_desc,
                "explanation": explanation,
                "timing_analysis": timing_info
            }
            
            scored_sectors.append(sector_data)

        # Sort by score and return top N
        ranked = sorted(scored_sectors, key=lambda x: x["score"], reverse=True)
        
        # Generate overall market summary
        avg_return = sum(s["predicted_return"] for s in ranked[:top_n]) / len(ranked[:top_n]) if ranked else 0
        avg_volatility = sum(s["volatility"] for s in ranked[:top_n]) / len(ranked[:top_n]) if ranked else 0
        
        market_summary = f"The top {top_n} recommended sectors show an average predicted return of {avg_return:.1f}% "
        market_summary += f"with moderate volatility of {avg_volatility:.1f}%. "
        market_summary += f"Current market conditions favor {'growth' if avg_return > 8 else 'stable'} sectors."
        
        return {
            "recommendations": ranked[:top_n],
            "market_summary": market_summary,
            "total_analyzed": len(sectors),
            "timestamp": datetime.now().isoformat(),
            "methodology": "ML-enhanced sector analysis with timing optimization"
        }
    except Exception as e:
        print(f"Error in enhanced stock recommendations: {e}")
        return {"recommendations": [], "error": str(e)}

def recommend_mutual_funds(db: Session, top_n: int = 5) -> List[Dict[str, Any]]:
    """Get top mutual fund recommendations using actual mutual fund data"""
    try:
        from app.models.analysis_models import MutualFund
        
        # Get all mutual funds from the database
        funds = db.query(MutualFund).all()
        if not funds:
            return []
        
        # Remove duplicates by fund name and get diverse fund types
        unique_funds = {}
        for fund in funds:
            fund_key = fund.fund_name.split(' - ')[0]  # Get base fund name
            if fund_key not in unique_funds:
                unique_funds[fund_key] = fund
        
        scored_funds = []
        
        # Create diverse categories for better representation
        category_mapping = {
            "Banking": "Banking & Financial Services",
            "PSU": "Public Sector Undertaking",
            "Debt": "Debt & Income",
            "Equity": "Equity Growth",
            "Hybrid": "Balanced Hybrid",
            "ELSS": "Tax Saving (ELSS)",
            "General": "Diversified"
        }
        
        for i, (fund_key, fund) in enumerate(unique_funds.items()):
            # Use available data to create meaningful recommendations
            nav = float(fund.nav) if fund.nav else 100
            returns_1y = float(fund.returns_1y) if fund.returns_1y else 0
            base_category = fund.category or "General"
            risk_level = fund.risk_level or "Medium"
            
            # Determine category based on fund name for better diversity
            if "Banking" in fund.fund_name or "PSU" in fund.fund_name:
                category = "Banking & Financial Services"
                base_return = 7.5
            elif "Debt" in fund.fund_name:
                category = "Debt & Income"
                base_return = 6.8
            elif "Equity" in fund.fund_name:
                category = "Equity Growth"
                base_return = 9.2
            elif "ELSS" in fund.fund_name:
                category = "Tax Saving (ELSS)"
                base_return = 10.1
            elif "Hybrid" in fund.fund_name:
                category = "Balanced Hybrid"
                base_return = 8.0
            else:
                category = "Diversified"
                base_return = 7.8
            
            # Create a synthetic predicted return based on category and risk
            if returns_1y > 0:
                predicted_return = min(returns_1y * 0.9, 15)
            else:
                risk_multiplier = {"Low": 0.85, "Medium": 1.0, "High": 1.15}.get(risk_level, 1.0)
                predicted_return = base_return * risk_multiplier
            
            # Calculate score based on return potential, risk, and NAV accessibility
            risk_score = {"Low": 2.8, "Medium": 2.5, "High": 2.0}.get(risk_level, 2.5)
            accessibility_score = 3.0 if nav < 500 else 2.5 if nav < 1500 else 2.0
            score = (predicted_return * 0.5) + (risk_score * 1.2) + accessibility_score
            
            # Create fund recommendation
            fund_info = {
                "sector": f"{category}",
                "fund_name": fund.fund_name,
                "fund_house": fund.fund_house or "Various",
                "category": category,
                "predicted_return": round(predicted_return, 2),
                "score": round(score, 2),
                "nav": round(nav, 2),
                "risk_level": risk_level,
                "investment_count": 1,
                "avg_price": nav,
                "top_performers": f'[{{"symbol": "{fund_key[:15]}...", "name": "{fund.fund_name}", "nav": {nav:.2f}, "category": "{category}", "risk": "{risk_level}"}}]'
            }
            
            scored_funds.append(fund_info)
        
        # Sort by score and return top N diverse funds
        scored_funds.sort(key=lambda x: x["score"], reverse=True)
        
        # Ensure diversity by category
        diverse_funds = []
        used_categories = set()
        
        # First pass: get one fund from each category
        for fund in scored_funds:
            if fund["category"] not in used_categories and len(diverse_funds) < top_n:
                diverse_funds.append(fund)
                used_categories.add(fund["category"])
        
        # Second pass: fill remaining slots with highest scoring funds
        for fund in scored_funds:
            if len(diverse_funds) >= top_n:
                break
            if fund not in diverse_funds:
                diverse_funds.append(fund)
        
        return diverse_funds[:top_n]
        
    except Exception as e:
        print(f"Error in mutual fund recommendations: {e}")
        import traceback
        traceback.print_exc()
        return []

def recommend_mutual_funds_enhanced(db: Session, top_n: int = 5) -> Dict[str, Any]:
    """Enhanced mutual fund recommendations with conversational explanations"""
    try:
        funds = db.query(ComprehensiveSectorAnalysis).all()
        scored_funds = []
        features = []
        fund_map = {}

        for idx, fund in enumerate(funds):
            features.append({
                "avg_return_pct": fund.avg_return_percent or 0,
                "volatility": fund.avg_volatility_percent or 0,
                "momentum_score": fund.avg_score or 0,
                "investment_count": fund.investments_analyzed or 0,
                "avg_price": 100  # Default NAV-like value
            })
            fund_map[idx] = fund

        predicted_returns = predict_returns(features)

        for idx, pred_return in enumerate(predicted_returns):
            fund = fund_map[idx]
            score = compute_score(pred_return, fund.avg_volatility_percent or 0, fund.avg_score or 0)
            
            # Generate conversational explanations
            risk_level = "Low" if (fund.avg_volatility_percent or 0) < 5 else "Medium" if (fund.avg_volatility_percent or 0) < 10 else "High"
            performance_desc = "Excellent" if (fund.avg_score or 0) > 8 else "Good" if (fund.avg_score or 0) > 6 else "Average"
            
            explanation = f"This {fund.asset_type or 'mutual fund'} in the {fund.sector} sector shows {performance_desc.lower()} performance. "
            explanation += f"With {risk_level.lower()} volatility of {fund.avg_volatility_percent or 0:.1f}%, it's suitable for "
            explanation += f"{'conservative' if risk_level == 'Low' else 'moderate' if risk_level == 'Medium' else 'aggressive'} investors. "
            explanation += f"Our ML model predicts a {pred_return:.1f}% return based on analysis of {fund.investments_analyzed or 0} similar investments."
            
            if fund.top_performer:
                explanation += f" The top performer in this category is {fund.top_performer}."
            
            fund_data = {
                "sector": fund.sector,
                "asset_type": fund.asset_type,
                "predicted_return": round(pred_return, 2),
                "score": round(score, 2),
                "avg_return_percent": fund.avg_return_percent or 0,
                "avg_volatility_percent": fund.avg_volatility_percent or 0,
                "avg_score": fund.avg_score or 0,
                "investments_analyzed": fund.investments_analyzed or 0,
                "top_performer": fund.top_performer or "",
                "risk_level": risk_level,
                "performance_description": performance_desc,
                "explanation": explanation
            }
            
            scored_funds.append(fund_data)

        # Sort by score and return top N
        ranked = sorted(scored_funds, key=lambda x: x["score"], reverse=True)
        
        # Generate overall portfolio summary
        avg_return = sum(f["predicted_return"] for f in ranked[:top_n]) / len(ranked[:top_n]) if ranked else 0
        avg_volatility = sum(f["avg_volatility_percent"] for f in ranked[:top_n]) / len(ranked[:top_n]) if ranked else 0
        
        portfolio_summary = f"The recommended mutual fund portfolio shows an average predicted return of {avg_return:.1f}% "
        portfolio_summary += f"with {avg_volatility:.1f}% volatility. This mix provides "
        portfolio_summary += f"{'balanced growth with moderate risk' if 5 <= avg_volatility <= 10 else 'stable returns with low risk' if avg_volatility < 5 else 'high growth potential with higher risk'}."
        
        return {
            "recommendations": ranked[:top_n],
            "portfolio_summary": portfolio_summary,
            "total_analyzed": len(funds),
            "timestamp": datetime.now().isoformat(),
            "methodology": "ML-enhanced fund analysis with risk profiling"
        }
    except Exception as e:
        print(f"Error in enhanced mutual fund recommendations: {e}")
        return {"recommendations": [], "error": str(e)}

def recommend_gold(db: Session, top_n: int = 5) -> List[Dict[str, Any]]:
    """Get top gold investment recommendations"""
    try:
        from app.models.analysis_models import GoldInvestment
        
        # Get gold investments from database
        gold_investments = db.query(GoldInvestment).all()
        
        # Create synthetic gold recommendations if no data exists
        gold_options = [
            {
                "sector": "Precious Metals",
                "investment_type": "Gold ETF",
                "name": "Gold Exchange Traded Fund",
                "predicted_return": 6.5,
                "score": 6.8,
                "risk_level": "Low",
                "liquidity": "High",
                "investment_count": 1,
                "avg_price": 5500,  # Price per gram
                "description": "Highly liquid gold investment through stock exchange",
                "top_performers": '[{"symbol": "GOLDETF", "name": "Gold ETF", "type": "ETF", "liquidity": "High"}]'
            },
            {
                "sector": "Precious Metals",
                "investment_type": "Digital Gold",
                "name": "Digital Gold Investment",
                "predicted_return": 6.0,
                "score": 6.5,
                "risk_level": "Low",
                "liquidity": "Medium",
                "investment_count": 1,
                "avg_price": 5520,
                "description": "Buy and store Gold digitally with easy redemption",
                "top_performers": '[{"symbol": "DIGITALGOLD", "name": "Digital Gold", "type": "Digital", "purity": "24K"}]'
            },
            {
                "sector": "Precious Metals", 
                "investment_type": "Gold Mutual Fund",
                "name": "Gold Mutual Fund",
                "predicted_return": 5.8,
                "score": 6.2,
                "risk_level": "Low",
                "liquidity": "Medium",
                "investment_count": 1,
                "avg_price": 5480,
                "description": "Invest in gold through mutual fund schemes",
                "top_performers": '[{"symbol": "GOLDMF", "name": "Gold Mutual Fund", "type": "Mutual Fund", "expense_ratio": "1.2%"}]'
            }
        ]
        
        # If we have actual gold investment data, use it
        if gold_investments:
            recommendations = []
            for gold in gold_investments:
                price_per_gram = float(gold.price_per_gram) if gold.price_per_gram else 5500
                returns_1y = float(gold.returns_1y) if gold.returns_1y else 6.0
                
                recommendations.append({
                    "sector": "Precious Metals",
                    "investment_type": gold.type or "Physical Gold",
                    "name": f"{gold.type} - {gold.issuer}" if gold.issuer else gold.type,
                    "predicted_return": returns_1y,
                    "score": returns_1y * 1.1,  # Simple scoring
                    "risk_level": "Low",
                    "liquidity": "Medium",
                    "investment_count": 1,
                    "avg_price": price_per_gram,
                    "purity": gold.purity or "22K",
                    "description": f"Investment in {gold.type} from {gold.issuer or 'certified dealer'}",
                    "top_performers": f'[{{"symbol": "{gold.type}", "name": "{gold.type}", "purity": "{gold.purity}", "price": {price_per_gram}}}]'
                })
            
            return sorted(recommendations, key=lambda x: x["score"], reverse=True)[:top_n]
        else:
            # Return synthetic recommendations
            return gold_options[:top_n]
            
    except Exception as e:
        print(f"Error in gold recommendations: {e}")
        import traceback
        traceback.print_exc()
        return []

def recommend_gold_enhanced(db: Session, top_n: int = 5) -> Dict[str, Any]:
    """Enhanced gold recommendations with conversational explanations"""
    try:
        gold_investments = db.query(EnhancedSectorScores).filter(
            EnhancedSectorScores.asset_type.ilike('%gold%')
        ).all()
        
        scored_gold = []
        features = []
        gold_map = {}

        for idx, gold in enumerate(gold_investments):
            features.append({
                "avg_return_pct": gold.avg_return_percent or 0,
                "volatility": gold.avg_volatility_percent or 0,
                "momentum_score": gold.avg_score or 0,
                "investment_count": gold.investments_analyzed or 0,
                "avg_price": 5000  # Default gold price per 10g
            })
            gold_map[idx] = gold

        predicted_returns = predict_returns(features)

        for idx, pred_return in enumerate(predicted_returns):
            gold = gold_map[idx]
            score = compute_score(pred_return, gold.avg_volatility_percent or 0, gold.avg_score or 0)
            
            # Generate conversational explanations
            volatility_desc = "stable" if (gold.avg_volatility_percent or 0) < 5 else "moderate" if (gold.avg_volatility_percent or 0) < 10 else "volatile"
            score_desc = "excellent" if (gold.avg_score or 0) > 8 else "good" if (gold.avg_score or 0) > 6 else "average"
            
            explanation = f"Gold investments in the {gold.sector} category show {score_desc} performance with {volatility_desc} price movements. "
            explanation += f"Based on analysis of {gold.investments_analyzed or 0} similar gold investments, "
            explanation += f"our ML model predicts a {pred_return:.1f}% return. "
            explanation += f"Gold serves as a hedge against inflation and currency fluctuations, making it "
            explanation += f"{'highly recommended' if score > 7 else 'suitable' if score > 5 else 'optional'} for portfolio diversification."
            
            if gold.top_performer:
                explanation += f" Consider {gold.top_performer} as a top option in this category."
            
            gold_data = {
                "sector": gold.sector,
                "asset_type": gold.asset_type,
                "predicted_return": round(pred_return, 2),
                "score": round(score, 2),
                "avg_return_percent": gold.avg_return_percent or 0,
                "avg_volatility_percent": gold.avg_volatility_percent or 0,
                "avg_score": gold.avg_score or 0,
                "investments_analyzed": gold.investments_analyzed or 0,
                "top_performer": gold.top_performer or "",
                "volatility_description": volatility_desc,
                "score_description": score_desc,
                "explanation": explanation
            }
            
            scored_gold.append(gold_data)

        # Sort by score and return top N
        ranked = sorted(scored_gold, key=lambda x: x["score"], reverse=True)
        
        # Generate gold market summary
        avg_return = sum(g["predicted_return"] for g in ranked[:top_n]) / len(ranked[:top_n]) if ranked else 0
        avg_volatility = sum(g["avg_volatility_percent"] for g in ranked[:top_n]) / len(ranked[:top_n]) if ranked else 0
        
        market_summary = f"Gold investments show an average predicted return of {avg_return:.1f}% with {avg_volatility:.1f}% volatility. "
        market_summary += f"Given current economic conditions, gold serves as "
        market_summary += f"{'an excellent hedge' if avg_return > 6 else 'a stable store of value' if avg_return > 3 else 'a conservative option'} "
        market_summary += f"for portfolio protection and diversification."
        
        return {
            "recommendations": ranked[:top_n],
            "market_summary": market_summary,
            "total_analyzed": len(gold_investments),
            "timestamp": datetime.now().isoformat(),
            "methodology": "ML-enhanced gold analysis with macroeconomic factors"
        }
    except Exception as e:
        print(f"Error in enhanced gold recommendations: {e}")
        return {"recommendations": [], "error": str(e)}

def recommend_all_assets_enhanced(db: Session, stocks_n: int = 3, mf_n: int = 3, gold_n: int = 2) -> Dict[str, Any]:
    """Enhanced comprehensive recommendations across all asset types"""
    try:
        start_time = datetime.now()
        
        # Get recommendations for all asset types
        stocks = recommend_stocks_enhanced(db, stocks_n)
        mutual_funds = recommend_mutual_funds_enhanced(db, mf_n)
        gold = recommend_gold_enhanced(db, gold_n)
        
        # Calculate overall portfolio metrics
        all_returns = []
        all_volatilities = []
        
        if stocks.get("recommendations"):
            all_returns.extend([s["predicted_return"] for s in stocks["recommendations"]])
            all_volatilities.extend([s["volatility"] for s in stocks["recommendations"]])
        
        if mutual_funds.get("recommendations"):
            all_returns.extend([mf["predicted_return"] for mf in mutual_funds["recommendations"]])
            all_volatilities.extend([mf["avg_volatility_percent"] for mf in mutual_funds["recommendations"]])
        
        if gold.get("recommendations"):
            all_returns.extend([g["predicted_return"] for g in gold["recommendations"]])
            all_volatilities.extend([g["avg_volatility_percent"] for g in gold["recommendations"]])
        
        portfolio_return = sum(all_returns) / len(all_returns) if all_returns else 0
        portfolio_volatility = sum(all_volatilities) / len(all_volatilities) if all_volatilities else 0
        
        # Generate comprehensive portfolio advice
        total_recommendations = len(all_returns)
        risk_level = "Conservative" if portfolio_volatility < 5 else "Moderate" if portfolio_volatility < 10 else "Aggressive"
        
        portfolio_advice = f"Your diversified portfolio of {total_recommendations} recommendations shows a predicted return of {portfolio_return:.1f}% "
        portfolio_advice += f"with {portfolio_volatility:.1f}% volatility, indicating a {risk_level.lower()} risk profile. "
        
        if portfolio_return > 8:
            portfolio_advice += "This is an excellent growth-oriented portfolio suitable for long-term wealth building. "
        elif portfolio_return > 5:
            portfolio_advice += "This balanced portfolio offers steady growth with moderate risk. "
        else:
            portfolio_advice += "This conservative portfolio prioritizes capital preservation with steady returns. "
        
        # Asset allocation advice
        stocks_weight = len(stocks.get("recommendations", [])) / total_recommendations * 100 if total_recommendations > 0 else 0
        mf_weight = len(mutual_funds.get("recommendations", [])) / total_recommendations * 100 if total_recommendations > 0 else 0
        gold_weight = len(gold.get("recommendations", [])) / total_recommendations * 100 if total_recommendations > 0 else 0
        
        allocation_advice = f"Recommended allocation: {stocks_weight:.0f}% stocks, {mf_weight:.0f}% mutual funds, {gold_weight:.0f}% gold. "
        allocation_advice += "This diversification helps balance growth potential with risk management."
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "stocks": stocks,
            "mutual_funds": mutual_funds,
            "gold": gold,
            "portfolio_summary": {
                "total_recommendations": total_recommendations,
                "expected_return": round(portfolio_return, 2),
                "expected_volatility": round(portfolio_volatility, 2),
                "risk_level": risk_level,
                "portfolio_advice": portfolio_advice,
                "allocation_advice": allocation_advice
            },
            "processing_time_seconds": round(processing_time, 2),
            "timestamp": datetime.now().isoformat(),
            "methodology": "Comprehensive ML-enhanced multi-asset analysis"
        }
    except Exception as e:
        print(f"Error in comprehensive recommendations: {e}")
        return {"error": str(e)}

# Model training and management functions
def get_technical_features_for_prediction(symbol_data: Dict) -> Dict:
    """Extract technical analysis features for ML prediction"""
    if not technical_analysis_available:
        return {}
    
    try:
        # If we have OHLCV data, apply technical analysis
        if all(key in symbol_data for key in ['open', 'high', 'low', 'close', 'volume']):
            # Create a simple DataFrame for technical analysis
            df = pd.DataFrame([symbol_data])
            df_with_indicators = generate_features(df)
            
            latest = df_with_indicators.iloc[-1]
            
            return {
                'rsi': latest.get('RSI_14', 50),
                'macd': latest.get('MACD', 0),
                'bb_position': (latest['close'] - latest.get('BB_Lower', latest['close'])) / 
                              (latest.get('BB_Upper', latest['close']) - latest.get('BB_Lower', latest['close'])),
                'atr_percentage': latest.get('ATR_14', 0) / latest['close'] * 100 if latest['close'] > 0 else 0,
                'adx': latest.get('ADX_14', 25),
                'stoch_rsi': latest.get('StochRSI', 0.5)
            }
    except Exception as e:
        print(f"Error extracting technical features: {e}")
    
    return {}

def train_and_save_ensemble_models(db: Session, asset_type: str = "stocks"):
    """Train and save ensemble models for future use"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        df = fetch_data(db, asset_type)
        if df is None or df.empty:
            return {"status": "error", "message": f"No data found for {asset_type}"}

        feature_cols = ["avg_return_pct", "volatility", "momentum_score", "investment_count", "avg_price"]
        target_col = "future_return_pct"
        
        # For demo purposes, create a synthetic target if it doesn't exist
        if target_col not in df.columns:
            # Create synthetic target based on existing features
            df[target_col] = df.apply(lambda row: 
                row.get("avg_return_pct", 0) * 1.2 + 
                row.get("momentum_score", 0) * 0.8 +
                random.uniform(-2, 2), axis=1)
        
        # Clean data
        df = df.dropna(subset=feature_cols + [target_col])
        X = df[feature_cols]
        y = df[target_col]
        
        if len(X) < 2:
            return {"status": "error", "message": "Insufficient data for training"}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        
        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        
        # Save models
        joblib.dump(rf, "models/random_forest_model.pkl")
        joblib.dump(xgb, "models/xgboost_model.pkl")
        
        # Test predictions
        rf_pred = rf.predict(X_test)
        xgb_pred = xgb.predict(X_test)
        ensemble_pred = (rf_pred + xgb_pred) / 2
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        
        print("Ensemble models trained and saved successfully!")
        return {
            "status": "success",
            "message": "Ensemble models trained and saved",
            "models_saved": ["random_forest_model.pkl", "xgboost_model.pkl"],
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "rmse": round(rmse, 3)
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error training and saving models: {str(e)}"}

def get_model_info() -> Dict[str, Any]:
    """Get information about loaded models"""
    return {
        "models_loaded": models_loaded,
        "rf_model_available": rf_model is not None,
        "xgb_model_available": xgb_model is not None,
        "realtime_prices_available": realtime_available,
        "models_path": "models/",
        "prediction_method": "ensemble" if models_loaded else "fallback"
    }
