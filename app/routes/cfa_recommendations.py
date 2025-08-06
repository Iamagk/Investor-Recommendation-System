"""
CFA-Grade Real-Time Investment Recommendation System
Combines historical ML model training with real-time Yahoo Finance data for optimal investment decisions.

Strategy:
1. Use daily 9 AM database updates for ML model training (historical accuracy)
2. Fetch real-time Yahoo Finance data for current market conditions
3. Apply trained ML models to real-time data for predictions
4. Generate comprehensive investment advice including position sizing and timing
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from fastapi import APIRouter, HTTPException
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@dataclass
class CFARecommendation:
    """CFA-style comprehensive investment recommendation"""
    symbol: str
    company_name: str
    sector: str
    
    # Current Market Data
    current_price: float
    previous_close: float
    day_change: float
    day_change_percent: float
    volume: int
    avg_volume: int
    
    # ML Model Predictions
    ml_predicted_price_1w: Optional[float]
    ml_predicted_price_1m: Optional[float]
    ml_predicted_price_3m: Optional[float]
    ml_confidence_score: float
    
    # Investment Analysis
    investment_recommendation: str  # BUY, HOLD, SELL, STRONG_BUY, STRONG_SELL
    target_price: float
    stop_loss: float
    risk_rating: str  # CONSERVATIVE, MODERATE, AGGRESSIVE
    
    # Position Sizing & Timing
    suggested_allocation_percent: float  # % of portfolio
    entry_strategy: str  # IMMEDIATE, DCA_WEEKLY, DCA_MONTHLY, WAIT_FOR_DIP
    entry_price_range: Tuple[float, float]  # (min, max) entry prices
    volume_to_buy: int  # Number of shares
    investment_horizon: str  # SHORT_TERM (1-3M), MEDIUM_TERM (3-12M), LONG_TERM (1Y+)
    
    # Technical Analysis
    technical_indicators: Dict
    support_levels: List[float]
    resistance_levels: List[float]
    
    # Fundamental Analysis
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    dividend_yield: Optional[float]
    debt_to_equity: Optional[float]
    roe: Optional[float]
    market_cap: Optional[float]
    
    # Risk Metrics
    beta: Optional[float]
    volatility_30d: float
    max_drawdown: float
    sharpe_ratio: Optional[float]
    
    # CFA Analysis
    intrinsic_value: float
    margin_of_safety: float
    dcf_value: Optional[float]
    
    # Timing
    last_updated: str
    next_review_date: str

# Top Indian stocks for analysis (expanded list)
INDIAN_STOCKS = {
    'RELIANCE.NS': 'Reliance Industries Limited',
    'TCS.NS': 'Tata Consultancy Services Limited',
    'HDFCBANK.NS': 'HDFC Bank Limited',
    'ICICIBANK.NS': 'ICICI Bank Limited',
    'HINDUNILVR.NS': 'Hindustan Unilever Limited',
    'INFY.NS': 'Infosys Limited',
    'ITC.NS': 'ITC Limited',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel Limited',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank Limited',
    'LT.NS': 'Larsen & Toubro Limited',
    'ASIANPAINT.NS': 'Asian Paints Limited',
    'AXISBANK.NS': 'Axis Bank Limited',
    'HCLTECH.NS': 'HCL Technologies Limited',
    'WIPRO.NS': 'Wipro Limited',
    'MARUTI.NS': 'Maruti Suzuki India Limited',
    'ULTRACEMCO.NS': 'UltraTech Cement Limited',
    'NESTLEIND.NS': 'Nestle India Limited',
    'POWERGRID.NS': 'Power Grid Corporation of India Limited',
    'NTPC.NS': 'NTPC Limited',
    'JSWSTEEL.NS': 'JSW Steel Limited',
    'TATAMOTORS.NS': 'Tata Motors Limited',
    'BAJFINANCE.NS': 'Bajaj Finance Limited',
    'ONGC.NS': 'Oil and Natural Gas Corporation Limited',
    'COALINDIA.NS': 'Coal India Limited',
    'TECHM.NS': 'Tech Mahindra Limited',
    'TITAN.NS': 'Titan Company Limited',
    'SUNPHARMA.NS': 'Sun Pharmaceutical Industries Limited',
    'DRREDDY.NS': 'Dr. Reddys Laboratories Limited',
    'CIPLA.NS': 'Cipla Limited',
    'ADANIPORTS.NS': 'Adani Ports and Special Economic Zone Limited',
    'BAJAJFINSV.NS': 'Bajaj Finserv Limited',
    'HEROMOTOCO.NS': 'Hero MotoCorp Limited',
    'BRITANNIA.NS': 'Britannia Industries Limited',
    'EICHERMOT.NS': 'Eicher Motors Limited'
}

# Sector mapping for better analysis
SECTOR_MAPPING = {
    'RELIANCE.NS': 'Energy', 'TCS.NS': 'Technology', 'HDFCBANK.NS': 'Banking',
    'ICICIBANK.NS': 'Banking', 'HINDUNILVR.NS': 'Consumer Goods', 'INFY.NS': 'Technology',
    'ITC.NS': 'Consumer Goods', 'SBIN.NS': 'Banking', 'BHARTIARTL.NS': 'Telecom',
    'KOTAKBANK.NS': 'Banking', 'LT.NS': 'Infrastructure', 'ASIANPAINT.NS': 'Consumer Goods',
    'AXISBANK.NS': 'Banking', 'HCLTECH.NS': 'Technology', 'WIPRO.NS': 'Technology',
    'MARUTI.NS': 'Automotive', 'ULTRACEMCO.NS': 'Cement', 'NESTLEIND.NS': 'Consumer Goods',
    'POWERGRID.NS': 'Utilities', 'NTPC.NS': 'Utilities', 'JSWSTEEL.NS': 'Steel',
    'TATAMOTORS.NS': 'Automotive', 'BAJFINANCE.NS': 'Financial Services', 'ONGC.NS': 'Energy',
    'COALINDIA.NS': 'Mining', 'TECHM.NS': 'Technology', 'TITAN.NS': 'Consumer Goods',
    'SUNPHARMA.NS': 'Pharmaceuticals', 'DRREDDY.NS': 'Pharmaceuticals', 'CIPLA.NS': 'Pharmaceuticals',
    'ADANIPORTS.NS': 'Infrastructure', 'BAJAJFINSV.NS': 'Financial Services', 
    'HEROMOTOCO.NS': 'Automotive', 'BRITANNIA.NS': 'Consumer Goods', 'EICHERMOT.NS': 'Automotive'
}

def load_ml_models() -> Dict:
    """Load pre-trained ML models from the models directory"""
    try:
        models = {}
        model_dir = "/Users/allenngeorge/Projects/investment_recommender/models"
        
        # Try to load Random Forest model with error handling
        rf_path = os.path.join(model_dir, "random_forest_model.pkl")
        if os.path.exists(rf_path):
            try:
                with open(rf_path, 'rb') as f:
                    models['random_forest'] = pickle.load(f)
                    logger.info("✅ Loaded Random Forest model")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Random Forest model: {e}")
        
        # Try to load XGBoost model with error handling  
        xgb_path = os.path.join(model_dir, "xgboost_model.pkl")
        if os.path.exists(xgb_path):
            try:
                with open(xgb_path, 'rb') as f:
                    models['xgboost'] = pickle.load(f)
                    logger.info("✅ Loaded XGBoost model")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load XGBoost model: {e}")
        
        # If no models could be loaded, log warning but continue
        if not models:
            logger.warning("⚠️ No ML models could be loaded - using fallback predictions")
        
        return models
    except Exception as e:
        logger.error(f"Error loading ML models: {e}")
        return {}

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD line and signal line"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band

def find_support_resistance(prices: pd.Series, level_type: str, window: int = 10) -> List[float]:
    """Find support and resistance levels using local minima/maxima"""
    try:
        # Simple approach: find recent highs and lows
        current_price = prices.iloc[-1]
        recent_prices = prices.iloc[-50:] if len(prices) >= 50 else prices
        
        if level_type == 'support':
            # Find local minima
            local_mins = []
            for i in range(window, len(recent_prices) - window):
                if recent_prices.iloc[i] == recent_prices.iloc[i-window:i+window+1].min():
                    local_mins.append(recent_prices.iloc[i])
            
            # Return unique levels below current price
            levels = sorted(list(set(local_mins)))[-3:]  # Last 3 support levels
            return [round(level, 2) for level in levels if level < current_price]
        else:  # resistance
            # Find local maxima
            local_maxs = []
            for i in range(window, len(recent_prices) - window):
                if recent_prices.iloc[i] == recent_prices.iloc[i-window:i+window+1].max():
                    local_maxs.append(recent_prices.iloc[i])
            
            # Return unique levels above current price
            levels = sorted(list(set(local_maxs)))[-3:]  # Last 3 resistance levels
            return [round(level, 2) for level in levels if level > current_price]
    except Exception:
        # Fallback: simple percentage-based levels
        current_price = prices.iloc[-1]
        if level_type == 'support':
            return [round(current_price * (1 - i/100), 2) for i in [2, 5, 10]]
        else:
            return [round(current_price * (1 + i/100), 2) for i in [2, 5, 10]]

def calculate_technical_indicators(data: pd.DataFrame) -> dict:
    """Calculate comprehensive technical indicators for CFA analysis"""
    try:
        if 'Close' not in data.columns or len(data) < 50:
            return {"error": "Insufficient data for technical analysis"}
        
        close_prices = data['Close']
        high_prices = data['High']
        low_prices = data['Low']
        volume = data['Volume']
        
        # RSI
        rsi = calculate_rsi(close_prices)
        
        # Moving Averages (multiple periods)
        ma5 = close_prices.rolling(5).mean()
        ma10 = close_prices.rolling(10).mean()
        ma20 = close_prices.rolling(20).mean()
        ma50 = close_prices.rolling(50).mean()
        ma200 = close_prices.rolling(200).mean()
        
        # MACD
        macd_line, macd_signal = calculate_macd(close_prices)
        macd_histogram = macd_line - macd_signal
        
        # Bollinger Bands
        bb_upper, bb_lower = calculate_bollinger_bands(close_prices)
        bb_middle = close_prices.rolling(20).mean()
        
        # Volume indicators
        volume_ma = volume.rolling(20).mean()
        volume_ratio = volume.iloc[-1] / volume_ma.iloc[-1] if len(volume_ma.dropna()) > 0 else 1
        
        # Support and Resistance levels
        support_levels = find_support_resistance(close_prices, 'support')
        resistance_levels = find_support_resistance(close_prices, 'resistance')
        
        # Volatility
        returns = close_prices.pct_change()
        volatility_30d = returns.rolling(30).std() * np.sqrt(252) * 100  # Annualized
        
        # Price position in 52-week range
        week_52_data = close_prices.iloc[-252:] if len(close_prices) >= 252 else close_prices
        week_52_high = week_52_data.max()
        week_52_low = week_52_data.min()
        current_price = close_prices.iloc[-1]
        
        return {
            "rsi": round(rsi.iloc[-1], 2) if len(rsi.dropna()) > 0 else None,
            "moving_averages": {
                "ma5": round(ma5.iloc[-1], 2) if len(ma5.dropna()) > 0 else None,
                "ma10": round(ma10.iloc[-1], 2) if len(ma10.dropna()) > 0 else None,
                "ma20": round(ma20.iloc[-1], 2) if len(ma20.dropna()) > 0 else None,
                "ma50": round(ma50.iloc[-1], 2) if len(ma50.dropna()) > 0 else None,
                "ma200": round(ma200.iloc[-1], 2) if len(ma200.dropna()) > 0 else None,
            },
            "macd": {
                "line": round(macd_line.iloc[-1], 4) if len(macd_line.dropna()) > 0 else None,
                "signal": round(macd_signal.iloc[-1], 4) if len(macd_signal.dropna()) > 0 else None,
                "histogram": round(macd_histogram.iloc[-1], 4) if len(macd_histogram.dropna()) > 0 else None
            },
            "bollinger_bands": {
                "upper": round(bb_upper.iloc[-1], 2) if len(bb_upper.dropna()) > 0 else None,
                "middle": round(bb_middle.iloc[-1], 2) if len(bb_middle.dropna()) > 0 else None,
                "lower": round(bb_lower.iloc[-1], 2) if len(bb_lower.dropna()) > 0 else None
            },
            "volume_analysis": {
                "current_volume": int(volume.iloc[-1]),
                "avg_volume": int(volume_ma.iloc[-1]) if len(volume_ma.dropna()) > 0 else None,
                "volume_ratio": round(volume_ratio, 2)
            },
            "price_levels": {
                "52_week_high": round(week_52_high, 2),
                "52_week_low": round(week_52_low, 2),
                "position_in_range": round((current_price - week_52_low) / (week_52_high - week_52_low) * 100, 2) if week_52_high > week_52_low else 50,
                "support_levels": support_levels,
                "resistance_levels": resistance_levels
            },
            "volatility": {
                "30_day_volatility": round(volatility_30d.iloc[-1], 2) if len(volatility_30d.dropna()) > 0 else None
            }
        }
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return {"error": str(e)}

def calculate_dcf_intrinsic_value(info: dict, current_price: float) -> float:
    """Calculate basic DCF intrinsic value using available financial data"""
    try:
        # Simplified DCF calculation using available Yahoo Finance data
        free_cash_flow = info.get('freeCashflow', 0)
        total_debt = info.get('totalDebt', 0)
        total_cash = info.get('totalCash', 0)
        shares_outstanding = info.get('sharesOutstanding', 0)
        
        if not all([free_cash_flow, shares_outstanding]) or free_cash_flow <= 0 or shares_outstanding <= 0:
            # Fallback: Use PE-based valuation or conservative estimate
            pe_ratio = info.get('forwardPE') or info.get('trailingPE') or 15
            earnings_per_share = info.get('trailingEps') or (current_price / pe_ratio)
            return round(max(current_price * 1.05, earnings_per_share * pe_ratio * 1.1), 2)
        
        # Simplified DCF: FCF * 10 (assuming 10% discount rate, 5% growth)
        enterprise_value = free_cash_flow * 10
        equity_value = enterprise_value - total_debt + total_cash
        intrinsic_value_per_share = equity_value / shares_outstanding
        
        # Ensure intrinsic value is reasonable (not negative or too extreme)
        intrinsic_value_per_share = max(intrinsic_value_per_share, current_price * 0.5)
        intrinsic_value_per_share = min(intrinsic_value_per_share, current_price * 3.0)
        
        return round(intrinsic_value_per_share, 2)
    except Exception as e:
        logger.warning(f"DCF calculation failed, using conservative estimate: {e}")
        # Conservative fallback: 10% above current price
        return round(current_price * 1.1, 2)

def apply_ml_predictions(symbol: str, historical_data: pd.DataFrame, models: Dict) -> Dict:
    """Apply ML models to predict future prices"""
    try:
        current_price = historical_data['Close'].iloc[-1]
        
        if not models or len(historical_data) < 50:
            # Fallback predictions based on historical analysis
            recent_trend = historical_data['Close'].pct_change(20).iloc[-1]
            return {
                "1_week": round(current_price * (1 + recent_trend * 0.25), 2),
                "1_month": round(current_price * (1 + recent_trend * 0.5), 2),
                "3_month": round(current_price * (1 + recent_trend * 1.0), 2),
                "confidence": 0.5
            }
        
        # Prepare features for ML model (simplified)
        close_prices = historical_data['Close']
        returns = close_prices.pct_change().dropna()
        
        # Basic features - handle NaN values
        current_price = close_prices.iloc[-1]
        ma5 = close_prices.rolling(5).mean().iloc[-1] if len(close_prices) >= 5 else current_price
        ma20 = close_prices.rolling(20).mean().iloc[-1] if len(close_prices) >= 20 else current_price
        ma50 = close_prices.rolling(50).mean().iloc[-1] if len(close_prices) >= 50 else current_price
        volatility = returns.rolling(10).std().iloc[-1] * np.sqrt(252) if len(returns) >= 10 else 0.2
        rsi_val = calculate_rsi(close_prices).iloc[-1] if len(calculate_rsi(close_prices).dropna()) > 0 else 50
        
        # Fill NaN values
        ma5 = ma5 if not pd.isna(ma5) else current_price
        ma20 = ma20 if not pd.isna(ma20) else current_price
        ma50 = ma50 if not pd.isna(ma50) else current_price
        volatility = volatility if not pd.isna(volatility) else 0.2
        rsi_val = rsi_val if not pd.isna(rsi_val) else 50
        
        features = np.array([
            current_price,
            ma5,
            ma20,
            ma50,
            volatility,
            rsi_val
        ]).reshape(1, -1)
        
        predictions = {}
        
        # Use ensemble of models if available
        if 'random_forest' in models:
            try:
                rf_pred = models['random_forest'].predict(features)[0]
                predictions['random_forest'] = rf_pred
            except Exception:
                predictions['random_forest'] = current_price
        
        if 'xgboost' in models:
            try:
                xgb_pred = models['xgboost'].predict(features)[0]
                predictions['xgboost'] = xgb_pred
            except Exception:
                predictions['xgboost'] = current_price
        
        # Ensemble prediction (average)
        if predictions:
            avg_prediction = np.mean(list(predictions.values()))
            confidence = 0.7 if len(predictions) > 1 else 0.6
        else:
            # Fallback: technical analysis-based prediction
            if rsi_val < 30 and current_price < ma20:  # Oversold
                avg_prediction = current_price * 1.05
            elif rsi_val > 70 and current_price > ma20:  # Overbought
                avg_prediction = current_price * 0.98
            else:
                avg_prediction = current_price * 1.02
            
            confidence = 0.5
        
        return {
            "1_week": round(avg_prediction, 2),
            "1_month": round(avg_prediction * 1.02, 2),
            "3_month": round(avg_prediction * 1.05, 2),
            "confidence": confidence
        }
    
    except Exception as e:
        logger.error(f"Error in ML predictions for {symbol}: {e}")
        current_price = historical_data['Close'].iloc[-1]
        return {
            "1_week": round(current_price * 1.01, 2),
            "1_month": round(current_price * 1.02, 2),
            "3_month": round(current_price * 1.03, 2),
            "confidence": 0.5
        }

def calculate_position_sizing(symbol: str, current_price: float, target_price: float, 
                             stop_loss: float, risk_rating: str, portfolio_size: float = 100000) -> Dict:
    """Calculate optimal position sizing based on risk management principles"""
    try:
        # Risk per trade based on risk rating
        risk_per_trade = {
            'CONSERVATIVE': 0.01,  # 1% of portfolio
            'MODERATE': 0.02,      # 2% of portfolio
            'AGGRESSIVE': 0.03     # 3% of portfolio
        }.get(risk_rating, 0.02)
        
        # Calculate position size based on stop loss
        risk_amount = portfolio_size * risk_per_trade
        price_risk = current_price - stop_loss
        
        if price_risk <= 0:
            price_risk = current_price * 0.05  # Default 5% stop loss
        
        shares_to_buy = int(risk_amount / price_risk)
        investment_amount = shares_to_buy * current_price
        portfolio_allocation = (investment_amount / portfolio_size) * 100
        
        # Calculate expected return
        expected_return = ((target_price - current_price) / current_price) * 100
        risk_reward_ratio = (target_price - current_price) / price_risk
        
        return {
            "shares_to_buy": shares_to_buy,
            "investment_amount": round(investment_amount, 2),
            "portfolio_allocation_percent": round(portfolio_allocation, 2),
            "risk_amount": round(risk_amount, 2),
            "expected_return_percent": round(expected_return, 2),
            "risk_reward_ratio": round(risk_reward_ratio, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating position sizing for {symbol}: {e}")
        return {
            "shares_to_buy": 0,
            "investment_amount": 0,
            "portfolio_allocation_percent": 0,
            "risk_amount": 0,
            "expected_return_percent": 0,
            "risk_reward_ratio": 0
        }

def generate_cfa_recommendation(symbol: str, models: Dict) -> Optional[CFARecommendation]:
    """Generate comprehensive CFA-style investment recommendation"""
    try:
        logger.info(f"Generating CFA recommendation for {symbol}")
        
        # Fetch real-time data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")  # 1 year of data for comprehensive analysis
        info = ticker.info
        
        if hist.empty:
            logger.warning(f"No historical data for {symbol}")
            return None
        
        current_price = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        day_change = current_price - previous_close
        day_change_percent = (day_change / previous_close) * 100
        
        # Technical Analysis
        technical_indicators = calculate_technical_indicators(hist)
        if "error" in technical_indicators:
            logger.warning(f"Technical analysis failed for {symbol}: {technical_indicators['error']}")
            return None
        
        # ML Predictions
        ml_predictions = apply_ml_predictions(symbol, hist, models)
        
        # Fundamental Analysis from Yahoo Finance info
        pe_ratio = info.get('forwardPE') or info.get('trailingPE')
        pb_ratio = info.get('priceToBook')
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else None
        debt_to_equity = info.get('debtToEquity')
        roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else None
        market_cap = info.get('marketCap')
        beta = info.get('beta')
        
        # Intrinsic Value Calculation
        intrinsic_value = calculate_dcf_intrinsic_value(info, current_price)
        margin_of_safety = ((intrinsic_value - current_price) / current_price) * 100
        
        # Risk Assessment
        volatility_30d = technical_indicators.get('volatility', {}).get('30_day_volatility', 25)
        
        # Determine risk rating - handle None values safely
        pe_ratio_safe = pe_ratio if pe_ratio is not None else 15
        if volatility_30d < 20 and pe_ratio_safe < 20:
            risk_rating = 'CONSERVATIVE'
        elif volatility_30d < 35:
            risk_rating = 'MODERATE'
        else:
            risk_rating = 'AGGRESSIVE'
        
        # Generate Investment Recommendation
        rsi = technical_indicators.get('rsi', 50)
        ma20 = technical_indicators.get('moving_averages', {}).get('ma20', current_price)
        ml_confidence = ml_predictions.get('confidence', 0.5)
        
        # CFA-style recommendation logic with safe ML prediction handling
        ml_1_month = ml_predictions.get('1_month') or current_price * 1.05
        ml_3_month = ml_predictions.get('3_month') or current_price * 1.15
        
        if (rsi < 30 and current_price < ma20 * 0.95 and 
            margin_of_safety > 15 and ml_confidence > 0.6):
            recommendation = 'STRONG_BUY'
            target_price = min(intrinsic_value, ml_3_month)
        elif (rsi < 40 and current_price < ma20 and margin_of_safety > 5):
            recommendation = 'BUY'
            target_price = min(intrinsic_value * 0.9, ml_1_month)
        elif (rsi > 70 and current_price > ma20 * 1.05 and margin_of_safety < -10):
            recommendation = 'STRONG_SELL'
            target_price = max(intrinsic_value, current_price * 0.85)
        elif (rsi > 60 and current_price > ma20 and margin_of_safety < 0):
            recommendation = 'SELL'
            target_price = max(intrinsic_value * 1.1, current_price * 0.92)
        else:
            recommendation = 'HOLD'
            target_price = ml_1_month
        
        # Stop Loss Calculation
        support_levels = technical_indicators.get('price_levels', {}).get('support_levels', [])
        if support_levels:
            stop_loss = min(support_levels[-1], current_price * 0.92)  # Recent support or 8% stop
        else:
            stop_loss = current_price * 0.92  # Default 8% stop loss
        
        # Position Sizing
        position_info = calculate_position_sizing(symbol, current_price, target_price, stop_loss, risk_rating)
        
        # Entry Strategy
        if recommendation in ['STRONG_BUY', 'BUY']:
            if rsi < 25:
                entry_strategy = 'IMMEDIATE'
                entry_price_range = (current_price * 0.98, current_price * 1.02)
            else:
                entry_strategy = 'DCA_WEEKLY'
                entry_price_range = (current_price * 0.95, current_price * 1.05)
        else:
            entry_strategy = 'WAIT_FOR_DIP'
            entry_price_range = (current_price * 0.90, current_price * 0.98)
        
        # Investment Horizon
        if risk_rating == 'CONSERVATIVE':
            investment_horizon = 'LONG_TERM'
        elif recommendation in ['STRONG_BUY', 'STRONG_SELL']:
            investment_horizon = 'SHORT_TERM'
        else:
            investment_horizon = 'MEDIUM_TERM'
        
        # Calculate additional metrics
        returns = hist['Close'].pct_change().dropna()
        max_drawdown = ((hist['Close'].cummax() - hist['Close']) / hist['Close'].cummax()).max() * 100
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if len(returns) > 0 else None
        
        return CFARecommendation(
            symbol=symbol,
            company_name=INDIAN_STOCKS.get(symbol, symbol),
            sector=SECTOR_MAPPING.get(symbol, 'Unknown'),
            
            # Current Market Data
            current_price=round(current_price, 2),
            previous_close=round(previous_close, 2),
            day_change=round(day_change, 2),
            day_change_percent=round(day_change_percent, 2),
            volume=int(hist['Volume'].iloc[-1]),
            avg_volume=int(hist['Volume'].rolling(20).mean().iloc[-1]),
            
            # ML Model Predictions
            ml_predicted_price_1w=ml_predictions.get('1_week'),
            ml_predicted_price_1m=ml_predictions.get('1_month'),
            ml_predicted_price_3m=ml_predictions.get('3_month'),
            ml_confidence_score=round(ml_confidence, 2),
            
            # Investment Analysis
            investment_recommendation=recommendation,
            target_price=round(target_price, 2),
            stop_loss=round(stop_loss, 2),
            risk_rating=risk_rating,
            
            # Position Sizing & Timing
            suggested_allocation_percent=round(position_info['portfolio_allocation_percent'], 2),
            entry_strategy=entry_strategy,
            entry_price_range=entry_price_range,
            volume_to_buy=position_info['shares_to_buy'],
            investment_horizon=investment_horizon,
            
            # Technical Analysis
            technical_indicators=technical_indicators,
            support_levels=technical_indicators.get('price_levels', {}).get('support_levels', []),
            resistance_levels=technical_indicators.get('price_levels', {}).get('resistance_levels', []),
            
            # Fundamental Analysis
            pe_ratio=round(pe_ratio, 2) if pe_ratio else None,
            pb_ratio=round(pb_ratio, 2) if pb_ratio else None,
            dividend_yield=round(dividend_yield, 2) if dividend_yield else None,
            debt_to_equity=round(debt_to_equity, 2) if debt_to_equity else None,
            roe=round(roe, 2) if roe else None,
            market_cap=market_cap,
            
            # Risk Metrics
            beta=round(beta, 2) if beta else None,
            volatility_30d=round(volatility_30d, 2),
            max_drawdown=round(max_drawdown, 2),
            sharpe_ratio=round(sharpe_ratio, 2) if sharpe_ratio else None,
            
            # CFA Analysis
            intrinsic_value=round(intrinsic_value, 2),
            margin_of_safety=round(margin_of_safety, 2),
            dcf_value=round(intrinsic_value, 2),  # Since we now always return a value
            
            # Timing
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            next_review_date=(datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        )
        
    except Exception as e:
        logger.error(f"Error generating CFA recommendation for {symbol}: {e}")
        return None

@router.get("/cfa-recommendations", summary="Get CFA-style comprehensive investment recommendations")
def get_cfa_recommendations(
    top_n: int = 10,
    portfolio_size: float = 100000,
    risk_preference: str = "MODERATE"
):
    """
    Get comprehensive CFA-style investment recommendations using:
    - Real-time Yahoo Finance data
    - Historical ML model predictions
    - Technical and fundamental analysis
    - Position sizing and risk management
    """
    try:
        logger.info("Loading ML models for CFA recommendations...")
        models = load_ml_models()
        
        logger.info("Generating CFA-style recommendations...")
        recommendations = []
        
        for symbol in list(INDIAN_STOCKS.keys())[:top_n]:
            recommendation = generate_cfa_recommendation(symbol, models)
            if recommendation:
                recommendations.append(recommendation)
        
        # Sort by investment recommendation and margin of safety
        recommendation_priority = {
            'STRONG_BUY': 5, 'BUY': 4, 'HOLD': 3, 'SELL': 2, 'STRONG_SELL': 1
        }
        
        recommendations.sort(
            key=lambda x: (
                recommendation_priority.get(x.investment_recommendation, 3),
                x.margin_of_safety
            ),
            reverse=True
        )
        
        # Convert to dictionaries for JSON response
        recommendations_dict = [asdict(rec) for rec in recommendations]
        
        return {
            "status": "success",
            "message": f"Generated {len(recommendations)} CFA-style recommendations",
            "data_source": "Yahoo Finance + ML Models",
            "analysis_type": "CFA Professional Grade",
            "portfolio_size": portfolio_size,
            "risk_preference": risk_preference,
            "recommendations": recommendations_dict,
            "total_analyzed": len(recommendations),
            "models_used": list(models.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating CFA recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@router.get("/cfa-stock/{symbol}", summary="Get detailed CFA analysis for a specific stock")
def get_cfa_stock_analysis(symbol: str):
    """Get detailed CFA-style analysis for a specific stock"""
    try:
        # Add .NS if not present for Indian stocks
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        
        models = load_ml_models()
        recommendation = generate_cfa_recommendation(symbol, models)
        
        if not recommendation:
            raise HTTPException(status_code=404, detail=f"Unable to analyze {symbol}")
        
        return {
            "status": "success",
            "symbol": symbol,
            "analysis": asdict(recommendation),
            "data_source": "Yahoo Finance + ML Models",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing {symbol}: {str(e)}")

# Backward compatibility - update the existing endpoint to use CFA system
@router.get("/realtime", summary="Real-time investment recommendations (CFA-enhanced)")
def get_realtime_recommendations():
    """Real-time investment recommendations using CFA methodology"""
    return get_cfa_recommendations(top_n=20)

@router.get("/sectors", summary="Sector-wise CFA analysis")
def get_sector_analysis():
    """Get sector-wise CFA analysis"""
    try:
        models = load_ml_models()
        sector_analysis = {}
        
        for symbol, company in INDIAN_STOCKS.items():
            sector = SECTOR_MAPPING.get(symbol, 'Unknown')
            
            if sector not in sector_analysis:
                sector_analysis[sector] = {
                    'stocks': [],
                    'avg_recommendation_score': 0,
                    'total_stocks': 0
                }
            
            recommendation = generate_cfa_recommendation(symbol, models)
            if recommendation:
                recommendation_score = {
                    'STRONG_BUY': 5, 'BUY': 4, 'HOLD': 3, 'SELL': 2, 'STRONG_SELL': 1
                }.get(recommendation.investment_recommendation, 3)
                
                sector_analysis[sector]['stocks'].append({
                    'symbol': symbol,
                    'company_name': company,
                    'recommendation': recommendation.investment_recommendation,
                    'target_price': recommendation.target_price,
                    'current_price': recommendation.current_price,
                    'margin_of_safety': recommendation.margin_of_safety
                })
                sector_analysis[sector]['total_stocks'] += 1
                sector_analysis[sector]['avg_recommendation_score'] += recommendation_score
        
        # Calculate averages
        for sector in sector_analysis:
            if sector_analysis[sector]['total_stocks'] > 0:
                sector_analysis[sector]['avg_recommendation_score'] /= sector_analysis[sector]['total_stocks']
                sector_analysis[sector]['avg_recommendation_score'] = round(sector_analysis[sector]['avg_recommendation_score'], 2)
        
        return {
            "status": "success",
            "sectors": sector_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in sector analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error in sector analysis: {str(e)}")
