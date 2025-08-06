from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import datetime
import pandas as pd
import yfinance as yf
import numpy as np
from app.db import get_db
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Daily Analysis"])

class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class TimeOfDay(Enum):
    PREMARKET = "PREMARKET"
    OPENING = "OPENING"
    MIDDAY = "MIDDAY"
    CLOSING = "CLOSING"
    AFTERMARKET = "AFTERMARKET"

@dataclass
class DayTradingSignal:
    symbol: str
    company_name: str
    signal: SignalType
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    expected_return: float
    risk_reward_ratio: float
    volume_strength: int
    time_sensitive: bool
    best_entry_time: str
    best_exit_time: str
    technical_reasons: List[str]

def get_current_time_session() -> TimeOfDay:
    """Determine current market session"""
    now = datetime.datetime.now()
    hour = now.hour
    
    if hour < 9:
        return TimeOfDay.PREMARKET
    elif 9 <= hour < 11:
        return TimeOfDay.OPENING
    elif 11 <= hour < 14:
        return TimeOfDay.MIDDAY
    elif 14 <= hour < 15:
        return TimeOfDay.CLOSING
    else:
        return TimeOfDay.AFTERMARKET

def fetch_enhanced_stock_data(symbols: List[str]) -> Dict[str, Dict]:
    """
    Fetch comprehensive real-time stock data for day trading analysis
    """
    stock_data = {}
    
    for symbol in symbols:
        try:
            yf_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
            ticker = yf.Ticker(yf_symbol)
            
            # Get different timeframes for comprehensive analysis
            hist_1d = ticker.history(period="1d", interval="5m")  # Intraday 5-minute data
            hist_5d = ticker.history(period="5d", interval="15m")  # 5-day 15-minute data
            hist_30d = ticker.history(period="30d")  # 30-day daily data
            
            info = ticker.info
            
            if len(hist_1d) > 0 and len(hist_30d) > 0:
                current_price = hist_1d['Close'].iloc[-1]
                
                # Calculate comprehensive technical indicators
                technical_indicators = calculate_enhanced_technical_indicators(
                    hist_1d, hist_5d, hist_30d
                )
                
                # Calculate volume analysis
                volume_analysis = calculate_volume_analysis(hist_1d, hist_5d)
                
                # Calculate price action patterns
                price_patterns = detect_price_patterns(hist_1d)
                
                # Calculate support and resistance levels
                support_resistance = calculate_support_resistance(hist_5d)
                
                stock_data[symbol] = {
                    'symbol': symbol,
                    'company_name': info.get('longName', symbol),
                    'current_price': float(current_price),
                    'previous_close': float(info.get('previousClose', current_price)),
                    'day_open': float(hist_1d['Open'].iloc[0]) if len(hist_1d) > 0 else float(current_price),
                    'day_high': float(hist_1d['High'].max()) if len(hist_1d) > 0 else float(current_price),
                    'day_low': float(hist_1d['Low'].min()) if len(hist_1d) > 0 else float(current_price),
                    'volume': int(hist_1d['Volume'].sum()) if len(hist_1d) > 0 else 0,
                    'avg_volume': int(hist_30d['Volume'].mean()),
                    'volume_ratio': float(hist_1d['Volume'].sum() / hist_30d['Volume'].mean()) if hist_30d['Volume'].mean() > 0 else 1.0,
                    'market_cap': info.get('marketCap', 0),
                    'sector': info.get('sector', 'Unknown'),
                    'technical_indicators': technical_indicators,
                    'volume_analysis': volume_analysis,
                    'price_patterns': price_patterns,
                    'support_resistance': support_resistance,
                    'intraday_prices': hist_1d['Close'].tolist()[-50:],  # Last 50 5-minute intervals
                    'intraday_times': [ts.strftime('%H:%M') for ts in hist_1d.index[-50:]]
                }
                
        except Exception as e:
            logger.error(f"Error fetching enhanced data for {symbol}: {str(e)}")
            continue
    
    return stock_data

def calculate_enhanced_technical_indicators(hist_1d, hist_5d, hist_30d) -> Dict[str, Any]:
    """
    Calculate comprehensive technical indicators for day trading
    """
    try:
        indicators = {}
        
        # Current day indicators (5-minute data)
        if len(hist_1d) > 0:
            close_1d = hist_1d['Close']
            
            # Short-term moving averages for intraday
            indicators['sma_5'] = float(close_1d.rolling(window=5).mean().iloc[-1])
            indicators['sma_10'] = float(close_1d.rolling(window=10).mean().iloc[-1])
            indicators['sma_20'] = float(close_1d.rolling(window=20).mean().iloc[-1])
            
            # Exponential moving averages
            indicators['ema_9'] = float(close_1d.ewm(span=9).mean().iloc[-1])
            indicators['ema_21'] = float(close_1d.ewm(span=21).mean().iloc[-1])
            
            # Intraday RSI (shorter period)
            delta = close_1d.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            
            # MACD for 5-minute intervals
            exp1 = close_1d.ewm(span=12).mean()
            exp2 = close_1d.ewm(span=26).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=9).mean()
            indicators['macd'] = float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0
            indicators['macd_signal'] = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
        # Multi-timeframe analysis
        if len(hist_30d) > 0:
            close_30d = hist_30d['Close']
            
            # Longer-term trend indicators
            indicators['sma_50'] = float(close_30d.rolling(window=20).mean().iloc[-1]) if len(close_30d) >= 20 else float(close_30d.iloc[-1])
            indicators['sma_200'] = float(close_30d.mean())  # Average as proxy for 200 SMA
            
            # Bollinger Bands (daily)
            rolling_mean = close_30d.rolling(window=20).mean()
            rolling_std = close_30d.rolling(window=20).std()
            indicators['bb_upper'] = float(rolling_mean.iloc[-1] + (rolling_std.iloc[-1] * 2))
            indicators['bb_lower'] = float(rolling_mean.iloc[-1] - (rolling_std.iloc[-1] * 2))
            indicators['bb_middle'] = float(rolling_mean.iloc[-1])
            
            # ATR (Average True Range) for volatility
            high_low = hist_30d['High'] - hist_30d['Low']
            high_close = np.abs(hist_30d['High'] - hist_30d['Close'].shift())
            low_close = np.abs(hist_30d['Low'] - hist_30d['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators['atr'] = float(true_range.rolling(window=14).mean().iloc[-1])
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculating enhanced technical indicators: {str(e)}")
        return {
            'sma_5': 100.0, 'sma_10': 100.0, 'sma_20': 100.0,
            'ema_9': 100.0, 'ema_21': 100.0, 'rsi': 50.0,
            'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
            'sma_50': 100.0, 'sma_200': 100.0,
            'bb_upper': 110.0, 'bb_lower': 90.0, 'bb_middle': 100.0,
            'atr': 5.0
        }

def calculate_volume_analysis(hist_1d, hist_5d) -> Dict[str, Any]:
    """
    Analyze volume patterns for day trading insights
    """
    try:
        volume_analysis = {}
        
        if len(hist_1d) > 0:
            current_volume = hist_1d['Volume'].sum()
            avg_volume_5d = hist_5d['Volume'].mean() if len(hist_5d) > 0 else current_volume
            
            volume_analysis['current_volume'] = int(current_volume)
            volume_analysis['avg_volume_5d'] = int(avg_volume_5d)
            volume_analysis['volume_ratio'] = float(current_volume / avg_volume_5d) if avg_volume_5d > 0 else 1.0
            
            # Volume trend analysis
            recent_volumes = hist_1d['Volume'].tolist()[-10:]  # Last 10 intervals
            if len(recent_volumes) >= 5:
                early_volume = sum(recent_volumes[:5])
                late_volume = sum(recent_volumes[5:])
                volume_analysis['volume_trend'] = 'INCREASING' if late_volume > early_volume else 'DECREASING'
            else:
                volume_analysis['volume_trend'] = 'STABLE'
            
            # Volume spikes
            volume_analysis['has_volume_spike'] = volume_analysis['volume_ratio'] > 1.5
            
        return volume_analysis
        
    except Exception as e:
        logger.error(f"Error calculating volume analysis: {str(e)}")
        return {
            'current_volume': 100000,
            'avg_volume_5d': 100000,
            'volume_ratio': 1.0,
            'volume_trend': 'STABLE',
            'has_volume_spike': False
        }

def detect_price_patterns(hist_1d) -> Dict[str, Any]:
    """
    Detect important price patterns for day trading
    """
    try:
        patterns = {}
        
        if len(hist_1d) >= 10:
            close_prices = hist_1d['Close']
            high_prices = hist_1d['High']
            low_prices = hist_1d['Low']
            
            # Recent price action
            recent_closes = close_prices.tolist()[-10:]
            
            # Trend detection
            if len(recent_closes) >= 5:
                early_avg = sum(recent_closes[:5]) / 5
                late_avg = sum(recent_closes[5:]) / 5
                
                if late_avg > early_avg * 1.02:
                    patterns['short_term_trend'] = 'BULLISH'
                elif late_avg < early_avg * 0.98:
                    patterns['short_term_trend'] = 'BEARISH'
                else:
                    patterns['short_term_trend'] = 'SIDEWAYS'
            
            # Support/Resistance breaks
            current_price = close_prices.iloc[-1]
            recent_high = high_prices[-10:].max()
            recent_low = low_prices[-10:].max()
            
            patterns['near_resistance'] = current_price > recent_high * 0.98
            patterns['near_support'] = current_price < recent_low * 1.02
            
            # Breakout detection
            volatility = close_prices.pct_change().std()
            recent_range = recent_high - recent_low
            patterns['potential_breakout'] = volatility > 0.02 and (patterns['near_resistance'] or patterns['near_support'])
            
        return patterns
        
    except Exception as e:
        logger.error(f"Error detecting price patterns: {str(e)}")
        return {
            'short_term_trend': 'SIDEWAYS',
            'near_resistance': False,
            'near_support': False,
            'potential_breakout': False
        }

def calculate_support_resistance(hist_5d) -> Dict[str, float]:
    """
    Calculate key support and resistance levels
    """
    try:
        if len(hist_5d) == 0:
            return {'support1': 100.0, 'support2': 95.0, 'resistance1': 105.0, 'resistance2': 110.0}
        
        highs = hist_5d['High']
        lows = hist_5d['Low']
        closes = hist_5d['Close']
        current_price = closes.iloc[-1]
        
        # Calculate support levels (recent lows)
        support_levels = []
        for i in range(1, len(lows) - 1):
            if lows.iloc[i] <= lows.iloc[i-1] and lows.iloc[i] <= lows.iloc[i+1]:
                support_levels.append(lows.iloc[i])
        
        # Calculate resistance levels (recent highs)
        resistance_levels = []
        for i in range(1, len(highs) - 1):
            if highs.iloc[i] >= highs.iloc[i-1] and highs.iloc[i] >= highs.iloc[i+1]:
                resistance_levels.append(highs.iloc[i])
        
        # Get closest levels
        support_levels = [s for s in support_levels if s < current_price]
        resistance_levels = [r for r in resistance_levels if r > current_price]
        
        support_levels.sort(reverse=True)  # Closest support first
        resistance_levels.sort()  # Closest resistance first
        
        return {
            'support1': float(support_levels[0]) if support_levels else float(current_price * 0.98),
            'support2': float(support_levels[1]) if len(support_levels) > 1 else float(current_price * 0.95),
            'resistance1': float(resistance_levels[0]) if resistance_levels else float(current_price * 1.02),
            'resistance2': float(resistance_levels[1]) if len(resistance_levels) > 1 else float(current_price * 1.05)
        }
        
    except Exception as e:
        logger.error(f"Error calculating support/resistance: {str(e)}")
        current_price = 100.0
        return {
            'support1': current_price * 0.98,
            'support2': current_price * 0.95,
            'resistance1': current_price * 1.02,
            'resistance2': current_price * 1.05
        }

def generate_daily_trading_signals(stock_data: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """
    Generate sophisticated daily trading signals for buy low, sell high strategy
    """
    signals = []
    current_session = get_current_time_session()
    
    for symbol, data in stock_data.items():
        try:
            current_price = data['current_price']
            day_open = data['day_open']
            day_high = data['day_high']
            day_low = data['day_low']
            technical = data['technical_indicators']
            volume = data['volume_analysis']
            patterns = data['price_patterns']
            sr_levels = data['support_resistance']
            
            # Calculate signal strength
            signal_score = 0
            technical_reasons = []
            
            # RSI Analysis (adjusted for day trading)
            rsi = technical['rsi']
            if rsi < 25:  # Oversold - strong buy signal
                signal_score += 3
                technical_reasons.append(f"Oversold RSI ({rsi:.1f})")
            elif rsi < 35:  # Mildly oversold
                signal_score += 2
                technical_reasons.append(f"Mildly oversold RSI ({rsi:.1f})")
            elif rsi > 75:  # Overbought - strong sell signal
                signal_score -= 3
                technical_reasons.append(f"Overbought RSI ({rsi:.1f})")
            elif rsi > 65:  # Mildly overbought
                signal_score -= 2
                technical_reasons.append(f"Mildly overbought RSI ({rsi:.1f})")
            
            # MACD Analysis
            macd = technical['macd']
            macd_signal = technical['macd_signal']
            macd_histogram = technical['macd_histogram']
            
            if macd > macd_signal and macd_histogram > 0:
                signal_score += 2
                technical_reasons.append("MACD bullish crossover")
            elif macd < macd_signal and macd_histogram < 0:
                signal_score -= 2
                technical_reasons.append("MACD bearish crossover")
            
            # Moving Average Analysis
            price_vs_sma5 = (current_price - technical['sma_5']) / technical['sma_5']
            price_vs_sma20 = (current_price - technical['sma_20']) / technical['sma_20']
            
            if current_price > technical['sma_5'] > technical['sma_20']:
                signal_score += 2
                technical_reasons.append("Price above rising MAs")
            elif current_price < technical['sma_5'] < technical['sma_20']:
                signal_score -= 2
                technical_reasons.append("Price below falling MAs")
            
            # Volume Analysis
            if volume['has_volume_spike'] and volume['volume_trend'] == 'INCREASING':
                signal_score += 1
                technical_reasons.append("High volume with increasing trend")
            elif volume['volume_ratio'] < 0.5:
                signal_score -= 1
                technical_reasons.append("Low volume warning")
            
            # Support/Resistance Analysis
            distance_to_support1 = (current_price - sr_levels['support1']) / current_price
            distance_to_resistance1 = (sr_levels['resistance1'] - current_price) / current_price
            
            if distance_to_support1 < 0.01:  # Very close to support
                signal_score += 2
                technical_reasons.append(f"Near key support (₹{sr_levels['support1']:.2f})")
            elif distance_to_resistance1 < 0.01:  # Very close to resistance
                signal_score -= 2
                technical_reasons.append(f"Near key resistance (₹{sr_levels['resistance1']:.2f})")
            
            # Pattern Analysis
            if patterns['potential_breakout']:
                if patterns['short_term_trend'] == 'BULLISH':
                    signal_score += 1
                    technical_reasons.append("Potential bullish breakout")
                elif patterns['short_term_trend'] == 'BEARISH':
                    signal_score -= 1
                    technical_reasons.append("Potential bearish breakdown")
            
            # Gap Analysis (comparing current price to day open)
            gap_percentage = (current_price - day_open) / day_open * 100
            if gap_percentage > 2:
                signal_score += 1
                technical_reasons.append(f"Positive gap ({gap_percentage:.1f}%)")
            elif gap_percentage < -2:
                signal_score -= 1
                technical_reasons.append(f"Negative gap ({gap_percentage:.1f}%)")
            
            # Time-based adjustments
            if current_session == TimeOfDay.OPENING:
                # Opening session - look for gap fills and momentum
                if abs(gap_percentage) > 1:
                    signal_score += 1 if gap_percentage > 0 else -1
            elif current_session == TimeOfDay.CLOSING:
                # Closing session - reduce position sizes, focus on quick moves
                signal_score = int(signal_score * 0.8)
            
            # Determine final signal
            if signal_score >= 4:
                signal = SignalType.STRONG_BUY
                confidence = min(95, 70 + signal_score * 3)
            elif signal_score >= 2:
                signal = SignalType.BUY
                confidence = min(85, 60 + signal_score * 4)
            elif signal_score <= -4:
                signal = SignalType.STRONG_SELL
                confidence = min(95, 70 + abs(signal_score) * 3)
            elif signal_score <= -2:
                signal = SignalType.SELL
                confidence = min(85, 60 + abs(signal_score) * 4)
            else:
                signal = SignalType.HOLD
                confidence = 50 + abs(signal_score) * 5
            
            # Calculate targets and stop loss
            atr = technical.get('atr', current_price * 0.02)
            
            if signal in [SignalType.STRONG_BUY, SignalType.BUY]:
                # For buy signals
                target_price = min(sr_levels['resistance1'], current_price + (atr * 1.5))
                stop_loss = max(sr_levels['support1'], current_price - (atr * 1.0))
                expected_return = ((target_price - current_price) / current_price) * 100
                
                # Optimal timing
                if current_session in [TimeOfDay.PREMARKET, TimeOfDay.OPENING]:
                    best_entry_time = "09:30-10:30 AM"
                    best_exit_time = "02:00-03:00 PM"
                else:
                    best_entry_time = "Now"
                    best_exit_time = "Before 03:00 PM"
                    
            elif signal in [SignalType.STRONG_SELL, SignalType.SELL]:
                # For sell signals (short selling or exit long positions)
                target_price = max(sr_levels['support1'], current_price - (atr * 1.5))
                stop_loss = min(sr_levels['resistance1'], current_price + (atr * 1.0))
                expected_return = ((current_price - target_price) / current_price) * 100
                
                best_entry_time = "Now"
                best_exit_time = "Before 03:00 PM"
                
            else:
                # Hold signal
                target_price = current_price * (1 + np.random.uniform(-0.01, 0.01))
                stop_loss = current_price * 0.98
                expected_return = 0.0
                best_entry_time = "Wait for better setup"
                best_exit_time = "N/A"
            
            # Risk-reward ratio
            risk = abs(current_price - stop_loss)
            reward = abs(target_price - current_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Volume strength (1-10 scale)
            volume_strength = min(10, max(1, int(volume['volume_ratio'] * 3)))
            
            trading_signal = {
                "symbol": symbol,
                "company_name": data['company_name'],
                "signal": signal.value,
                "signal_strength": abs(signal_score),
                "confidence": round(confidence, 1),
                "current_price": round(current_price, 2),
                "entry_price": round(current_price, 2),
                "target_price": round(target_price, 2),
                "stop_loss": round(stop_loss, 2),
                "expected_return": round(expected_return, 2),
                "risk_reward_ratio": round(risk_reward_ratio, 2),
                "volume_strength": volume_strength,
                "time_sensitive": signal != SignalType.HOLD,
                "best_entry_time": best_entry_time,
                "best_exit_time": best_exit_time,
                "technical_reasons": technical_reasons,
                "day_performance": {
                    "open": round(day_open, 2),
                    "high": round(day_high, 2),
                    "low": round(day_low, 2),
                    "current": round(current_price, 2),
                    "gap_percent": round(gap_percentage, 2)
                },
                "support_resistance": {
                    "support1": round(sr_levels['support1'], 2),
                    "resistance1": round(sr_levels['resistance1'], 2)
                },
                "volume_analysis": volume,
                "market_session": current_session.value,
                "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            signals.append(trading_signal)
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            continue
    
    # Sort by signal strength and confidence
    signals.sort(key=lambda x: (x['signal_strength'], x['confidence']), reverse=True)
    
    return signals

@router.get("/daily-signals")
async def get_daily_trading_signals(
    top_n: int = Query(default=15, description="Number of top signals to return"),
    signal_type: Optional[str] = Query(default=None, description="Filter by signal type: BUY, SELL, STRONG_BUY, STRONG_SELL"),
    min_confidence: float = Query(default=60.0, description="Minimum confidence level"),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive daily trading signals for buy low, sell high strategy
    """
    try:
        # Enhanced stock list for day trading
        day_trading_stocks = [
            "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "HINDUNILVR",
            "INFY", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
            "LT", "ASIANPAINT", "MARUTI", "AXISBANK", "HCLTECH",
            "WIPRO", "ULTRACEMCO", "NESTLEIND", "POWERGRID", "NTPC",
            "ONGC", "TATAMOTORS", "SUNPHARMA", "BAJFINANCE", "TECHM",
            "TITAN", "ADANIPORTS", "COALINDIA", "DRREDDY", "JSWSTEEL",
            "HEROMOTOCO", "BRITANNIA", "DIVISLAB", "EICHERMOT", "GRASIM",
            "HINDALCO", "INDUSINDBK", "BAJAJFINSV", "BAJAJ-AUTO", "CIPLA"
        ]
        
        logger.info("Fetching enhanced stock data for daily analysis...")
        stock_data = fetch_enhanced_stock_data(day_trading_stocks)
        
        if not stock_data:
            raise HTTPException(status_code=500, detail="Failed to fetch stock data")
        
        logger.info("Generating daily trading signals...")
        signals = generate_daily_trading_signals(stock_data)
        
        # Filter signals based on parameters
        if signal_type:
            signals = [s for s in signals if s['signal'] == signal_type.upper()]
        
        signals = [s for s in signals if s['confidence'] >= min_confidence]
        
        # Get top N signals
        top_signals = signals[:top_n]
        
        # Calculate market summary
        buy_signals = len([s for s in signals if s['signal'] in ['BUY', 'STRONG_BUY']])
        sell_signals = len([s for s in signals if s['signal'] in ['SELL', 'STRONG_SELL']])
        hold_signals = len([s for s in signals if s['signal'] == 'HOLD'])
        
        current_session = get_current_time_session()
        
        result = {
            "status": "success",
            "signals": top_signals,
            "market_summary": {
                "total_analyzed": len(stock_data),
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "hold_signals": hold_signals,
                "market_session": current_session.value,
                "best_trading_time": "09:30-11:00 AM and 02:00-03:00 PM",
                "market_sentiment": "BULLISH" if buy_signals > sell_signals else "BEARISH" if sell_signals > buy_signals else "NEUTRAL"
            },
            "trading_tips": [
                "🎯 Focus on stocks with risk-reward ratio > 2.0",
                "📊 High volume strength (>7) indicates strong momentum",
                "⏰ Best entry times are during opening and closing sessions", 
                "🛡️ Always use stop-loss orders for risk management",
                "💰 Take partial profits at resistance levels",
                "📈 Monitor 5-minute charts for precise entry/exit points"
            ],
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": "Yahoo Finance Real-time + Enhanced Technical Analysis"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in get_daily_trading_signals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating daily signals: {str(e)}")

@router.get("/market-scanner")
async def get_market_scanner(
    scan_type: str = Query(default="breakouts", description="Scanner type: breakouts, oversold, overbought, high_volume"),
    db: Session = Depends(get_db)
):
    """
    Advanced market scanner for specific trading opportunities
    """
    try:
        # Larger stock universe for scanning
        scanner_stocks = [
            "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "HINDUNILVR", "INFY", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
            "LT", "ASIANPAINT", "MARUTI", "AXISBANK", "HCLTECH", "WIPRO", "ULTRACEMCO", "NESTLEIND", "POWERGRID", "NTPC",
            "ONGC", "TATAMOTORS", "SUNPHARMA", "BAJFINANCE", "TECHM", "TITAN", "ADANIPORTS", "COALINDIA", "DRREDDY", "JSWSTEEL",
            "HEROMOTOCO", "BRITANNIA", "DIVISLAB", "EICHERMOT", "GRASIM", "HINDALCO", "INDUSINDBK", "BAJAJFINSV", "BAJAJ-AUTO", "CIPLA",
            "APOLLOHOSP", "HDFCLIFE", "SBILIFE", "ICICIPRULI", "VEDL", "GODREJCP", "PIDILITIND", "DABUR", "MARICO", "COLPAL"
        ]
        
        stock_data = fetch_enhanced_stock_data(scanner_stocks)
        
        if not stock_data:
            raise HTTPException(status_code=404, detail="No stock data available")
        
        filtered_stocks = []
        
        for symbol, data in stock_data.items():
            current_price = data['current_price']
            technical = data['technical_indicators']
            volume = data['volume_analysis']
            patterns = data['price_patterns']
            sr_levels = data['support_resistance']
            
            match_criteria = False
            scanner_reason = ""
            
            if scan_type == "breakouts":
                # Look for stocks breaking resistance with volume
                near_resistance = current_price >= sr_levels['resistance1'] * 0.99
                high_volume = volume['volume_ratio'] > 1.3
                bullish_momentum = technical['rsi'] > 50 and technical['macd'] > technical['macd_signal']
                
                if near_resistance and high_volume and bullish_momentum:
                    match_criteria = True
                    scanner_reason = f"Resistance breakout at ₹{sr_levels['resistance1']:.2f} with {volume['volume_ratio']:.1f}x volume"
                    
            elif scan_type == "oversold":
                # Look for oversold stocks near support
                oversold = technical['rsi'] < 35
                near_support = current_price <= sr_levels['support1'] * 1.02
                not_falling_knife = technical['macd_histogram'] > -0.5  # Not in severe downtrend
                
                if oversold and near_support and not_falling_knife:
                    match_criteria = True
                    scanner_reason = f"Oversold (RSI: {technical['rsi']:.1f}) near support ₹{sr_levels['support1']:.2f}"
                    
            elif scan_type == "overbought":
                # Look for overbought stocks near resistance
                overbought = technical['rsi'] > 70
                near_resistance = current_price >= sr_levels['resistance1'] * 0.98
                weakening = technical['macd_histogram'] < 0  # MACD showing weakness
                
                if overbought and near_resistance and weakening:
                    match_criteria = True
                    scanner_reason = f"Overbought (RSI: {technical['rsi']:.1f}) near resistance ₹{sr_levels['resistance1']:.2f}"
                    
            elif scan_type == "high_volume":
                # Look for unusual volume spikes
                volume_spike = volume['volume_ratio'] > 2.0
                price_movement = abs(data['current_price'] - data['day_open']) / data['day_open'] * 100 > 1
                
                if volume_spike and price_movement:
                    match_criteria = True
                    scanner_reason = f"Volume spike: {volume['volume_ratio']:.1f}x normal volume"
            
            if match_criteria:
                filtered_stocks.append({
                    "symbol": symbol,
                    "company_name": data['company_name'],
                    "current_price": round(current_price, 2),
                    "day_change_percent": round((current_price - data['day_open']) / data['day_open'] * 100, 2),
                    "volume_ratio": round(volume['volume_ratio'], 1),
                    "rsi": round(technical['rsi'], 1),
                    "scanner_reason": scanner_reason,
                    "support": round(sr_levels['support1'], 2),
                    "resistance": round(sr_levels['resistance1'], 2),
                    "sector": data.get('sector', 'Unknown')
                })
        
        # Sort by volume ratio for most active stocks
        filtered_stocks.sort(key=lambda x: x['volume_ratio'], reverse=True)
        
        return {
            "status": "success",
            "scan_type": scan_type,
            "results": filtered_stocks[:20],  # Top 20 matches
            "total_scanned": len(stock_data),
            "matches_found": len(filtered_stocks),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        logger.error(f"Error in market scanner: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running market scanner: {str(e)}")

@router.get("/position-calculator")
async def calculate_position_size(
    capital: float = Query(description="Available capital for trading"),
    risk_percent: float = Query(default=2.0, description="Risk percentage per trade (1-5%)"),
    entry_price: float = Query(description="Planned entry price"),
    stop_loss: float = Query(description="Stop loss price"),
    db: Session = Depends(get_db)
):
    """
    Calculate optimal position size for day trading based on risk management
    """
    try:
        if risk_percent > 5.0:
            raise HTTPException(status_code=400, detail="Risk percentage should not exceed 5% per trade")
        
        if entry_price <= stop_loss:
            raise HTTPException(status_code=400, detail="Entry price must be higher than stop loss for long positions")
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        # Calculate maximum risk amount
        max_risk_amount = capital * (risk_percent / 100)
        
        # Calculate position size
        position_size = int(max_risk_amount / risk_per_share)
        
        # Calculate total position value
        position_value = position_size * entry_price
        
        # Calculate percentage of capital used
        capital_used_percent = (position_value / capital) * 100
        
        # Position size recommendations
        if position_size == 0:
            recommendation = "⚠️ Risk is too high - consider adjusting stop loss or reducing risk percentage"
            max_safe_position = int(capital * 0.1 / entry_price)  # 10% of capital as emergency position
        elif capital_used_percent > 80:
            recommendation = "⚠️ High capital utilization - consider reducing position size"
            max_safe_position = int(capital * 0.5 / entry_price)
        elif capital_used_percent > 50:
            recommendation = "✅ Moderate position size - good for experienced traders"
            max_safe_position = position_size
        else:
            recommendation = "✅ Conservative position size - good for beginners"
            max_safe_position = position_size
        
        return {
            "status": "success",
            "position_calculation": {
                "recommended_quantity": position_size,
                "max_safe_quantity": max_safe_position,
                "position_value": round(position_value, 2),
                "capital_used_percent": round(capital_used_percent, 1),
                "risk_amount": round(max_risk_amount, 2),
                "risk_per_share": round(risk_per_share, 2),
                "recommendation": recommendation
            },
            "risk_management": {
                "max_loss_if_stopped": round(max_risk_amount, 2),
                "remaining_capital": round(capital - position_value, 2),
                "suggested_targets": [
                    round(entry_price + (risk_per_share * 1.5), 2),  # 1.5:1 reward
                    round(entry_price + (risk_per_share * 2.0), 2),  # 2:1 reward
                    round(entry_price + (risk_per_share * 3.0), 2)   # 3:1 reward
                ]
            },
            "trading_rules": [
                f"Never risk more than {risk_percent}% per trade",
                "Always use stop-loss orders",
                "Take partial profits at resistance levels",
                "Don't trade more than 3-5 positions simultaneously",
                "Cut losses quickly, let profits run"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error calculating position size: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in position calculation: {str(e)}")
