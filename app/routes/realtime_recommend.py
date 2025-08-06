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
from dataclasses import dataclass
import logging
from fastapi import APIRouter, HTTPException
import pickle
import os
from sqlalchemy.orm import Session
from app.db import get_db
from fastapi import Depends

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
    ml_predicted_price_1w: float
    ml_predicted_price_1m: float
    ml_predicted_price_3m: float
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

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Real-time Recommendations"])

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
        
        # Load Random Forest model
        rf_path = os.path.join(model_dir, "random_forest_model.pkl")
        if os.path.exists(rf_path):
            with open(rf_path, 'rb') as f:
                models['random_forest'] = pickle.load(f)
                logger.info("✅ Loaded Random Forest model")
        
        # Load XGBoost model
        xgb_path = os.path.join(model_dir, "xgboost_model.pkl")
        if os.path.exists(xgb_path):
            with open(xgb_path, 'rb') as f:
                models['xgboost'] = pickle.load(f)
                logger.info("✅ Loaded XGBoost model")
        
        return models
    except Exception as e:
        logger.error(f"Error loading ML models: {e}")
        return {}

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

@dataclass
class RealtimeStock:
    symbol: str
    company_name: str
    current_price: float
    previous_close: float
    day_change: float
    day_change_percent: float
    volume: int
    market_cap: float
    sector: str
    pe_ratio: float
    pb_ratio: float
    dividend_yield: float
    fifty_two_week_high: float
    fifty_two_week_low: float
    beta: float
    price_history: List[float]
    dates: List[str]
    technical_indicators: Dict
    investment_score: float
    recommendation: str
    target_price: float
    risk_level: str
    investment_horizon: str
    expected_annual_return: float
    last_updated: str

def fetch_realtime_investment_data():
    """Fetch real-time data for investment analysis from Yahoo Finance"""
    logger.info("Fetching real-time investment data from Yahoo Finance...")
    
    recommendations = []
    failed_stocks = []
    
    for symbol in REALTIME_STOCKS:
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            
            # Get historical data (3 months for better analysis)
            hist = ticker.history(period="3mo", interval="1d")
            
            if hist.empty or len(hist) < 30:
                failed_stocks.append(symbol)
                continue
                
            # Get current price data
            current_price = hist['Close'].iloc[-1]
            previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            volume = hist['Volume'].iloc[-1]
            
            # Calculate day change
            day_change = current_price - previous_close
            day_change_percent = (day_change / previous_close) * 100 if previous_close > 0 else 0
            
            # Calculate technical indicators for investment analysis
            technical_indicators = calculate_technical_indicators(hist)
            
            # Calculate investment score and recommendation
            investment_score, recommendation, target_price, risk_level = calculate_investment_recommendation(
                hist, technical_indicators, info
            )
            
            # Extract price history for charts (last 30 days)
            price_history = hist['Close'].tail(30).tolist()
            dates = [date.strftime('%Y-%m-%d') for date in hist.index.tail(30)]
            
            stock_rec = RealtimeStock(
                symbol=symbol.replace('.NS', ''),
                company_name=info.get('longName', symbol.replace('.NS', '')),
                current_price=float(current_price),
                previous_close=float(previous_close),
                day_change=float(day_change),
                day_change_percent=float(day_change_percent),
                volume=int(volume),
                market_cap=info.get('marketCap', 0),
                sector=info.get('sector', 'Unknown'),
                pe_ratio=info.get('trailingPE', 0),
                pb_ratio=info.get('priceToBook', 0),
                dividend_yield=info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                fifty_two_week_high=info.get('fiftyTwoWeekHigh', current_price),
                fifty_two_week_low=info.get('fiftyTwoWeekLow', current_price),
                beta=info.get('beta', 1.0),
                price_history=price_history,
                dates=dates,
                technical_indicators=technical_indicators,
                investment_score=investment_score,
                recommendation=recommendation,
                target_price=target_price,
                risk_level=risk_level,
                investment_horizon=determine_investment_horizon(technical_indicators, risk_level),
                expected_annual_return=calculate_expected_return(
                    current_price, target_price, technical_indicators
                ),
                last_updated=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            recommendations.append(stock_rec)
            
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {str(e)}")
            failed_stocks.append(symbol)
            continue
    
    # Sort by investment score (highest first)
    recommendations.sort(key=lambda x: x.investment_score, reverse=True)
    
    logger.info(f"Successfully processed {len(recommendations)} stocks, failed: {len(failed_stocks)}")
    return recommendations

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD line and signal line."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band

def calculate_technical_indicators(data: pd.DataFrame) -> dict:
    """Calculate technical indicators for the given price data."""
    try:
        # Ensure we have Close prices
        if 'Close' not in data.columns:
            return {"error": "No Close price data available"}
        
        close_prices = data['Close']
        
        # Check if we have enough data
        if len(close_prices) < 14:
            return {"error": "Insufficient data for technical analysis"}
        
        # RSI
        rsi = calculate_rsi(close_prices)
        
        # Moving Averages
        ma20 = close_prices.rolling(20).mean()
        ma50 = close_prices.rolling(50).mean()
        ma200 = close_prices.rolling(200).mean()
        
        # MACD
        macd_line, macd_signal = calculate_macd(close_prices)
        
        # Bollinger Bands
        bb_upper, bb_lower = calculate_bollinger_bands(close_prices)
        
        # Current values (most recent)
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else None
        current_price = close_prices.iloc[-1]
        current_ma20 = ma20.iloc[-1] if len(ma20.dropna()) > 0 else None
        current_ma50 = ma50.iloc[-1] if len(ma50.dropna()) > 0 else None
        current_ma200 = ma200.iloc[-1] if len(ma200.dropna()) > 0 else None
        
        # 52-week high/low - fix the tail issue
        week_52_data = close_prices.iloc[-252:] if len(close_prices) >= 252 else close_prices
        week_52_high = week_52_data.max()
        week_52_low = week_52_data.min()
        
        return {
            "rsi": round(current_rsi, 2) if current_rsi else None,
            "ma20": round(current_ma20, 2) if current_ma20 else None,
            "ma50": round(current_ma50, 2) if current_ma50 else None,
            "ma200": round(current_ma200, 2) if current_ma200 else None,
            "macd": {
                "line": round(macd_line.iloc[-1], 2) if len(macd_line) > 0 and not pd.isna(macd_line.iloc[-1]) else None,
                "signal": round(macd_signal.iloc[-1], 2) if len(macd_signal) > 0 and not pd.isna(macd_signal.iloc[-1]) else None
            },
            "bollinger_bands": {
                "upper": round(bb_upper.iloc[-1], 2) if len(bb_upper) > 0 and not pd.isna(bb_upper.iloc[-1]) else None,
                "lower": round(bb_lower.iloc[-1], 2) if len(bb_lower) > 0 and not pd.isna(bb_lower.iloc[-1]) else None
            },
            "52_week_high": round(week_52_high, 2),
            "52_week_low": round(week_52_low, 2),
            "position_in_52w_range": round((current_price - week_52_low) / (week_52_high - week_52_low) * 100, 2) if week_52_high > week_52_low else 50
        }
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return {"error": str(e)}

def calculate_investment_recommendation(hist_data, indicators, info):
    """Calculate investment recommendation based on fundamental and technical analysis"""
    close_prices = hist_data['Close']
    current_price = close_prices.iloc[-1]
    
    # Initialize scoring
    score = 50  # Start with neutral score
    
    # Technical Analysis Scoring (40% weight)
    # Trend Analysis
    if indicators['moving_average_20'] > indicators['moving_average_50']:
        score += 8  # Short-term uptrend
    if indicators['moving_average_50'] > indicators['moving_average_200']:
        score += 10  # Long-term uptrend
    
    # RSI Analysis (for entry timing)
    if 30 <= indicators['rsi'] <= 70:
        score += 5  # Neutral RSI is good for investment
    elif indicators['rsi'] < 30:
        score += 8  # Oversold, good entry point
    elif indicators['rsi'] > 80:
        score -= 5  # Overbought, wait for correction
    
    # MACD Trend
    if indicators['macd'] > 0:
        score += 6  # Positive momentum
    
    # Price position in 52-week range
    if 20 <= indicators['price_52w_position'] <= 60:
        score += 8  # Good entry zone
    elif indicators['price_52w_position'] < 20:
        score += 12  # Near 52-week low, potential value
    
    # Fundamental Analysis Scoring (40% weight)
    pe_ratio = info.get('trailingPE', 25)
    pb_ratio = info.get('priceToBook', 3)
    dividend_yield = info.get('dividendYield', 0)
    
    # P/E Ratio scoring
    if 10 <= pe_ratio <= 20:
        score += 10  # Reasonable valuation
    elif pe_ratio < 10:
        score += 15  # Potentially undervalued
    elif pe_ratio > 30:
        score -= 8  # Expensive
    
    # P/B Ratio scoring
    if pb_ratio < 2:
        score += 8  # Good book value
    elif pb_ratio > 5:
        score -= 5  # Expensive relative to book value
    
    # Dividend yield
    if dividend_yield and dividend_yield > 0.02:  # >2%
        score += 5  # Good dividend yield
    
    # Risk Analysis (20% weight)
    beta = info.get('beta', 1.0)
    if 0.8 <= beta <= 1.2:
        score += 5  # Moderate risk
    elif beta > 1.5:
        score -= 8  # High risk
        
    # Volatility penalty
    if indicators['volatility'] > 40:
        score -= 10  # High volatility penalty
    elif indicators['volatility'] < 20:
        score += 5  # Low volatility bonus
    
    # Ensure score is in valid range
    score = max(0, min(100, score))
    
    # Determine recommendation
    if score >= 75:
        recommendation = "STRONG BUY"
        risk_level = "LOW" if indicators['volatility'] < 25 else "MODERATE"
    elif score >= 60:
        recommendation = "BUY"
        risk_level = "MODERATE"
    elif score >= 40:
        recommendation = "HOLD"
        risk_level = "MODERATE" if indicators['volatility'] < 30 else "HIGH"
    else:
        recommendation = "AVOID"
        risk_level = "HIGH"
    
    # Calculate target price (12-month target)
    growth_factor = 1 + (score - 50) / 200  # Score-based growth expectation
    target_price = current_price * growth_factor
    
    return score, recommendation, target_price, risk_level

def determine_investment_horizon(indicators, risk_level):
    """Determine recommended investment horizon"""
    if risk_level == "LOW" and indicators['volatility'] < 20:
        return "Long-term (3-5 years)"
    elif risk_level == "MODERATE":
        return "Medium-term (1-3 years)"
    else:
        return "Short-term (6-12 months)"

def calculate_expected_return(current_price, target_price, indicators):
    """Calculate expected annual return"""
    if current_price <= 0:
        return 0
    
    # 12-month return expectation
    return ((target_price - current_price) / current_price) * 100

@router.get("/realtime", summary="Get real-time investment recommendations with Yahoo Finance data")
def get_realtime_recommendations(
    top_n: int = Query(default=20, description="Number of top recommendations to return"),
    min_score: float = Query(default=50, description="Minimum investment score (0-100)"),
    risk_filter: str = Query(default="ALL", description="Risk filter: LOW, MODERATE, HIGH, ALL"),
    sector_filter: str = Query(default="", description="Comma-separated sectors to filter")
):
    """
    Get real-time investment recommendations using live Yahoo Finance data
    
    Features:
    - Live market data from Yahoo Finance
    - Technical and fundamental analysis
    - Investment scoring (0-100)
    - Risk assessment
    - 12-month target prices
    - Investment horizon recommendations
    """
    try:
        logger.info("Generating real-time investment recommendations...")
        
        # Fetch real-time data
        recommendations = fetch_realtime_investment_data()
        
        if not recommendations:
            return {
                "status": "warning",
                "message": "No recommendations available",
                "recommendations": [],
                "total_analyzed": 0,
                "market_status": "UNKNOWN",
                "last_updated": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "data_source": "Yahoo Finance (Real-time)"
            }
        
        # Apply filters
        filtered_recs = recommendations
        
        # Score filter
        if min_score > 0:
            filtered_recs = [r for r in filtered_recs if r.investment_score >= min_score]
        
        # Risk filter
        if risk_filter != "ALL":
            filtered_recs = [r for r in filtered_recs if r.risk_level == risk_filter]
        
        # Sector filter
        if sector_filter:
            sectors = [s.strip().upper() for s in sector_filter.split(',')]
            filtered_recs = [r for r in filtered_recs if r.sector.upper() in sectors]
        
        # Limit results
        filtered_recs = filtered_recs[:top_n]
        
        # Convert to response format
        response_data = []
        for rec in filtered_recs:
            response_data.append({
                "symbol": rec.symbol,
                "company_name": rec.company_name,
                "current_price": rec.current_price,
                "previous_close": rec.previous_close,
                "day_change": rec.day_change,
                "day_change_percent": rec.day_change_percent,
                "volume": rec.volume,
                "market_cap": rec.market_cap,
                "sector": rec.sector,
                "pe_ratio": rec.pe_ratio,
                "pb_ratio": rec.pb_ratio,
                "dividend_yield": rec.dividend_yield,
                "fifty_two_week_high": rec.fifty_two_week_high,
                "fifty_two_week_low": rec.fifty_two_week_low,
                "beta": rec.beta,
                "price_history": rec.price_history,
                "dates": rec.dates,
                "technical_indicators": rec.technical_indicators,
                "investment_score": rec.investment_score,
                "recommendation": rec.recommendation,
                "target_price": rec.target_price,
                "risk_level": rec.risk_level,
                "investment_horizon": rec.investment_horizon,
                "expected_annual_return": rec.expected_annual_return,
                "last_updated": rec.last_updated
            })
        
        # Determine market status (simple check)
        current_hour = datetime.datetime.now().hour
        market_status = "OPEN" if 9 <= current_hour <= 15 else "CLOSED"
        
        return {
            "status": "success",
            "message": f"Real-time investment recommendations generated successfully",
            "recommendations": response_data,
            "total_analyzed": len(recommendations),
            "filtered_results": len(response_data),
            "market_status": market_status,
            "last_updated": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "data_source": "Yahoo Finance (Real-time)",
            "filters_applied": {
                "min_score": min_score,
                "risk_filter": risk_filter,
                "sector_filter": sector_filter,
                "top_n": top_n
            }
        }
        
    except Exception as e:
        logger.error(f"Error in get_realtime_recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating real-time recommendations: {str(e)}")

@router.get("/realtime/sectors", summary="Get sector-wise real-time investment analysis")
def get_realtime_sector_analysis():
    """Get sector-wise breakdown of real-time investment opportunities"""
    try:
        recommendations = fetch_realtime_investment_data()
        
        # Group by sector
        sectors = {}
        for rec in recommendations:
            sector = rec.sector or "Unknown"
            if sector not in sectors:
                sectors[sector] = {
                    "sector_name": sector,
                    "stocks_count": 0,
                    "avg_score": 0,
                    "avg_expected_return": 0,
                    "top_stocks": [],
                    "sector_recommendation": "HOLD"
                }
            
            sectors[sector]["stocks_count"] += 1
            sectors[sector]["top_stocks"].append({
                "symbol": rec.symbol,
                "company_name": rec.company_name,
                "investment_score": rec.investment_score,
                "recommendation": rec.recommendation,
                "expected_annual_return": rec.expected_annual_return
            })
        
        # Calculate sector averages and recommendations
        for sector_data in sectors.values():
            if sector_data["stocks_count"] > 0:
                scores = [stock["investment_score"] for stock in sector_data["top_stocks"]]
                returns = [stock["expected_annual_return"] for stock in sector_data["top_stocks"]]
                
                sector_data["avg_score"] = sum(scores) / len(scores)
                sector_data["avg_expected_return"] = sum(returns) / len(returns)
                
                # Sort stocks by score
                sector_data["top_stocks"].sort(key=lambda x: x["investment_score"], reverse=True)
                sector_data["top_stocks"] = sector_data["top_stocks"][:3]  # Top 3 per sector
                
                # Sector recommendation
                if sector_data["avg_score"] >= 70:
                    sector_data["sector_recommendation"] = "OVERWEIGHT"
                elif sector_data["avg_score"] >= 55:
                    sector_data["sector_recommendation"] = "NEUTRAL"
                else:
                    sector_data["sector_recommendation"] = "UNDERWEIGHT"
        
        # Sort sectors by average score
        sorted_sectors = sorted(sectors.values(), key=lambda x: x["avg_score"], reverse=True)
        
        return {
            "status": "success",
            "message": "Sector-wise real-time analysis completed",
            "sectors": sorted_sectors,
            "total_sectors": len(sectors),
            "last_updated": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "data_source": "Yahoo Finance (Real-time)"
        }
        
    except Exception as e:
        logger.error(f"Error in get_realtime_sector_analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating sector analysis: {str(e)}")
