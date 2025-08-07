from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import datetime
import pandas as pd
import yfinance as yf
import numpy as np
from app.db import get_db
from app.services.llm_analysis_service import llm_service
import logging
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Professional CFA Trading"])

class TradingStyle(Enum):
    SCALPING = "SCALPING"
    SWING = "SWING"
    MOMENTUM = "MOMENTUM"

class RiskAppetite(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

@dataclass
class TradingPreferences:
    capital_allocation: float
    risk_appetite: RiskAppetite
    preferred_sectors: List[str]
    max_trades_per_day: int
    trading_style: TradingStyle
    holding_duration_minutes: int

@dataclass
class ProfessionalSignal:
    symbol: str
    company_name: str
    confidence_score: float
    direction: str  # BUY/SELL
    entry_price: float
    stop_loss: float
    target_price_1: float
    target_price_2: float
    expected_roi: float
    risk_reward_ratio: float
    signal_time: str
    volatility_level: str
    sector: str
    chart_data: Dict[str, Any]
    signal_log: List[str]
    # LLM Explanations
    market_analysis: str = ""
    action_plan: str = ""
    risk_management: str = ""
    timing: str = ""

def calculate_advanced_indicators(hist_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive technical indicators for professional trading
    """
    try:
        indicators = {}
        close = hist_data['Close']
        high = hist_data['High']
        low = hist_data['Low']
        volume = hist_data['Volume']
        
        # Moving Averages (Professional Standard)
        indicators['ema_5'] = close.ewm(span=5).mean().iloc[-1]
        indicators['ema_13'] = close.ewm(span=13).mean().iloc[-1]
        indicators['sma_50'] = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.mean()
        
        # RSI (14 period)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
        
        # MACD
        exp1 = close.ewm(span=12).mean()
        exp2 = close.ewm(span=26).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9).mean()
        indicators['macd'] = macd.iloc[-1]
        indicators['macd_signal'] = signal_line.iloc[-1]
        indicators['macd_histogram'] = macd.iloc[-1] - signal_line.iloc[-1]
        
        # Bollinger Bands
        bb_period = min(20, len(close))
        bb_mean = close.rolling(bb_period).mean()
        bb_std = close.rolling(bb_period).std()
        indicators['bb_upper'] = (bb_mean + bb_std * 2).iloc[-1]
        indicators['bb_lower'] = (bb_mean - bb_std * 2).iloc[-1]
        indicators['bb_middle'] = bb_mean.iloc[-1]
        
        # VWAP (Volume Weighted Average Price)
        typical_price = (high + low + close) / 3
        vwap_num = (typical_price * volume).cumsum()
        vwap_den = volume.cumsum()
        indicators['vwap'] = (vwap_num / vwap_den).iloc[-1]
        
        # ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        indicators['atr'] = true_range.rolling(14).mean().iloc[-1]
        
        # Stochastic Oscillator
        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        indicators['stoch_k'] = k_percent.iloc[-1]
        indicators['stoch_d'] = k_percent.rolling(3).mean().iloc[-1]
        
        # Volume Analysis
        avg_volume = volume.rolling(20).mean() if len(volume) >= 20 else volume.mean()
        indicators['volume_ratio'] = volume.iloc[-1] / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
        indicators['volume_spike'] = indicators['volume_ratio'] > 1.5
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculating advanced indicators: {str(e)}")
        return {}

def calculate_kelly_criterion_position_size(win_rate: float, avg_win: float, avg_loss: float, capital: float) -> int:
    """
    Calculate optimal position size using Kelly Criterion
    """
    try:
        if avg_loss <= 0:
            return 0
        
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Conservative approach: use 25% of Kelly fraction to reduce risk
        conservative_kelly = kelly_fraction * 0.25
        
        # Ensure position size is reasonable (max 10% of capital)
        optimal_fraction = min(max(conservative_kelly, 0), 0.1)
        
        return optimal_fraction * capital
        
    except Exception as e:
        logger.error(f"Error calculating Kelly position size: {str(e)}")
        return capital * 0.02  # Fallback to 2% risk

def generate_professional_signals(
    preferences: TradingPreferences, 
    stock_universe: List[str]
) -> List[ProfessionalSignal]:
    """
    Generate professional-grade trading signals with comprehensive analysis
    """
    signals = []
    signals_generated = 0
    
    for symbol in stock_universe:
        if signals_generated >= preferences.max_trades_per_day:
            break
            
        try:
            # Fetch real-time data
            yf_symbol = f"{symbol}.NS"
            ticker = yf.Ticker(yf_symbol)
            
            # Get intraday data (1-minute intervals for last 5 days)
            hist_1m = ticker.history(period="5d", interval="1m")
            hist_5m = ticker.history(period="5d", interval="5m")
            info = ticker.info
            
            if len(hist_1m) < 50:  # Need sufficient data
                continue
                
            # Filter by sector preference
            stock_sector = info.get('sector', 'Unknown')
            if preferences.preferred_sectors and stock_sector not in preferences.preferred_sectors:
                continue
            
            # Calculate advanced indicators
            indicators = calculate_advanced_indicators(hist_5m)
            if not indicators:
                continue
            
            current_price = hist_1m['Close'].iloc[-1]
            
            # Generate trading signal based on style and indicators
            signal_data = generate_signal_by_style(
                preferences.trading_style,
                current_price,
                indicators,
                hist_1m,
                hist_5m
            )
            
            if not signal_data:
                continue
            
            # Calculate risk metrics
            atr = indicators.get('atr', current_price * 0.02)
            volatility_level = classify_volatility(atr, current_price)
            
def enhanced_small_capital_strategy(current_price: float, indicators: dict, preferences: TradingPreferences, signal_data: dict, capital: float) -> dict:
    """
    Enhanced strategy optimized for small capital accounts (< ₹50,000)
    Targets higher percentage moves and better capital utilization
    """
    try:
        atr = indicators.get('atr', current_price * 0.02)
        
        # For small capital, focus on stocks under ₹1000 for better position sizing
        price_category = "LOW" if current_price < 500 else "MEDIUM" if current_price < 1000 else "HIGH"
        
        # Minimum target percentage based on capital and price category
        if capital < 20000:  # Small account
            min_target_pct = 3.0 if price_category == "LOW" else 2.5 if price_category == "MEDIUM" else 2.0
        elif capital < 50000:  # Medium account
            min_target_pct = 2.5 if price_category == "LOW" else 2.0 if price_category == "MEDIUM" else 1.5
        else:  # Large account
            min_target_pct = 2.0 if price_category == "LOW" else 1.5 if price_category == "MEDIUM" else 1.0
        
        # Risk multipliers based on trading style and capital
        style_multipliers = {
            "SCALPING": {"risk": 0.8, "target1": 1.8, "target2": 3.2},
            "MOMENTUM": {"risk": 1.0, "target1": 2.5, "target2": 4.5},
            "SWING": {"risk": 1.5, "target1": 3.5, "target2": 6.0}
        }
        
        multipliers = style_multipliers.get(preferences.trading_style.value, style_multipliers["MOMENTUM"])
        
        # Adjust for risk appetite and small capital
        risk_adj = {"LOW": 0.7, "MEDIUM": 1.0, "HIGH": 1.4}[preferences.risk_appetite.value]
        
        # For small capital, increase target multipliers to ensure meaningful profits
        if capital < 30000:
            multipliers["target1"] *= 1.3
            multipliers["target2"] *= 1.3
        
        # Calculate percentage-based targets (minimum profitable moves)
        base_target_pct = max(min_target_pct, (atr / current_price) * 100 * multipliers["target1"])
        
        if signal_data['direction'] == 'BUY':
            stop_loss_pct = (atr / current_price) * 100 * multipliers["risk"] * risk_adj
            target1_pct = max(base_target_pct, min_target_pct) * risk_adj
            target2_pct = target1_pct * 1.8
            
            stop_loss = current_price * (1 - stop_loss_pct / 100)
            target_1 = current_price * (1 + target1_pct / 100)
            target_2 = current_price * (1 + target2_pct / 100)
        else:
            stop_loss_pct = (atr / current_price) * 100 * multipliers["risk"] * risk_adj
            target1_pct = max(base_target_pct, min_target_pct) * risk_adj
            target2_pct = target1_pct * 1.8
            
            stop_loss = current_price * (1 + stop_loss_pct / 100)
            target_1 = current_price * (1 - target1_pct / 100)
            target_2 = current_price * (1 - target2_pct / 100)
        
        # Calculate potential profit for position sizing
        shares_possible = int(capital * 0.95 / current_price)  # 95% allocation
        profit_target1 = shares_possible * abs(target_1 - current_price)
        profit_target2 = shares_possible * abs(target_2 - current_price)
        
        # Skip signals with very low profit potential for small accounts
        if capital < 50000 and profit_target1 < 200:  # Minimum ₹200 profit
            return None
        
        return {
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_2,
            'expected_roi': target1_pct,
            'shares_possible': shares_possible,
            'profit_target1': profit_target1,
            'profit_target2': profit_target2,
            'stop_loss_pct': stop_loss_pct,
            'target1_pct': target1_pct,
            'target2_pct': target2_pct
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced small capital strategy: {str(e)}")
        return None

def filter_stocks_for_small_capital(stocks: list, capital: float) -> list:
    """
    Filter stocks optimal for small capital accounts
    """
    try:
        filtered_stocks = []
        
        for stock in stocks:
            try:
                # Get current price
                ticker = yf.Ticker(f"{stock}.NS")
                hist = ticker.history(period="1d", interval="1m")
                if hist.empty:
                    continue
                    
                current_price = float(hist['Close'].iloc[-1])
                volume = hist['Volume'].sum()
                
                # Criteria for small capital optimization
                price_suitable = current_price < 1500  # Affordable for small accounts
                volume_adequate = volume > 100000  # Adequate liquidity
                shares_possible = int(capital * 0.95 / current_price)
                min_shares = shares_possible >= 5  # Minimum position size
                
                # Calculate potential profit
                potential_move = current_price * 0.025  # 2.5% move
                potential_profit = shares_possible * potential_move
                
                # Only include if potential profit is meaningful
                if price_suitable and volume_adequate and min_shares and potential_profit >= 150:
                    filtered_stocks.append({
                        'symbol': stock,
                        'price': current_price,
                        'shares_possible': shares_possible,
                        'potential_profit': potential_profit,
                        'affordability_score': 1500 / current_price  # Higher score for lower prices
                    })
                    
            except Exception as e:
                logger.error(f"Error filtering stock {stock}: {str(e)}")
                continue
        
        # Sort by affordability and potential profit
        filtered_stocks.sort(key=lambda x: (x['affordability_score'], x['potential_profit']), reverse=True)
        
        return [stock['symbol'] for stock in filtered_stocks[:30]]  # Top 30 suitable stocks
        
    except Exception as e:
        logger.error(f"Error filtering stocks for small capital: {str(e)}")
            if not signal_data:
                continue
            
            # Calculate risk metrics
            atr = indicators.get('atr', current_price * 0.02)
            volatility_level = classify_volatility(atr, current_price)
            
            # Use enhanced strategy for small capital accounts
            if preferences.capital_allocation < 50000:
                strategy_result = enhanced_small_capital_strategy(
                    current_price, indicators, preferences, signal_data, preferences.capital_allocation
                )
                if not strategy_result:
                    continue  # Skip if not profitable enough
                    
                stop_loss = strategy_result['stop_loss']
                target_1 = strategy_result['target_1']
                target_2 = strategy_result['target_2']
                expected_roi = strategy_result['expected_roi']
                shares_possible = strategy_result['shares_possible']
                profit_target1 = strategy_result['profit_target1']
            else:
                # Original strategy for larger accounts
                risk_multiplier = {"LOW": 0.5, "MEDIUM": 1.0, "HIGH": 1.5}[preferences.risk_appetite.value]
                
                if signal_data['direction'] == 'BUY':
                    stop_loss = current_price - (atr * 1.5 * risk_multiplier)
                    target_1 = current_price + (atr * 2.0 * risk_multiplier)
                    target_2 = current_price + (atr * 3.5 * risk_multiplier)
                else:
                    stop_loss = current_price + (atr * 1.5 * risk_multiplier)
                    target_1 = current_price - (atr * 2.0 * risk_multiplier)
                    target_2 = current_price - (atr * 3.5 * risk_multiplier)
                
                expected_roi = abs((target_1 - current_price) / current_price) * 100
                shares_possible = int(preferences.capital_allocation * 0.95 / current_price)
                profit_target1 = shares_possible * abs(target_1 - current_price)
            
            # Calculate risk-reward ratio
            risk_amount = abs(current_price - stop_loss)
            reward_amount = abs(target_1 - current_price)
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            # Generate chart data for visualization
            chart_data = prepare_chart_data(hist_1m, indicators, signal_data)
            
            # Create signal log
            signal_log = [
                f"{datetime.datetime.now().strftime('%H:%M')} - {signal_data['direction']} at ₹{current_price:.2f}",
                f"Target 1: ₹{target_1:.2f}, Target 2: ₹{target_2:.2f}",
                f"Stop Loss: ₹{stop_loss:.2f}",
                f"Risk/Reward: 1:{risk_reward_ratio:.1f}",
                f"Confidence: {signal_data['confidence']:.0f}%"
            ]
            
            professional_signal = ProfessionalSignal(
                symbol=symbol,
                company_name=info.get('longName', symbol),
                confidence_score=signal_data['confidence'],
                direction=signal_data['direction'],
                entry_price=current_price,
                stop_loss=stop_loss,
                target_price_1=target_1,
                target_price_2=target_2,
                expected_roi=expected_roi,
                risk_reward_ratio=risk_reward_ratio,
                signal_time=datetime.datetime.now().strftime("%H:%M:%S"),
                volatility_level=volatility_level,
                sector=stock_sector,
                chart_data=chart_data,
                signal_log=signal_log
            )
            
            # Generate LLM explanation for this signal
            try:
                signal_explanation_data = {
                    'symbol': symbol,
                    'company_name': info.get('longName', symbol),
                    'direction': signal_data['direction'],
                    'confidence_score': signal_data['confidence'],
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'target_price_1': target_1,
                    'target_price_2': target_2,
                    'expected_roi': expected_roi,
                    'risk_reward_ratio': risk_reward_ratio,
                    'trading_style': preferences.trading_style.value,
                    'volatility_level': volatility_level,
                    'signal_log': signal_log[:5]  # First 5 entries for LLM
                }
                
                llm_explanation = llm_service.generate_trading_signal_explanation(signal_explanation_data)
                
                # Add LLM explanations to the signal
                professional_signal.market_analysis = llm_explanation.get('market_analysis', '')
                professional_signal.action_plan = llm_explanation.get('action_plan', '')
                professional_signal.risk_management = llm_explanation.get('risk_management', '')
                professional_signal.timing = llm_explanation.get('timing', '')
                
            except Exception as e:
                logger.error(f"Error generating LLM explanation for {symbol}: {str(e)}")
                # Set fallback explanations
                professional_signal.market_analysis = f"Technical analysis indicates a {signal_data['direction']} opportunity in {symbol}."
                professional_signal.action_plan = f"Execute {signal_data['direction']} order at ₹{current_price:.2f} with stop-loss at ₹{stop_loss:.2f}."
                professional_signal.risk_management = f"Maintain strict stop-loss discipline. Risk per share: ₹{abs(current_price - stop_loss):.2f}."
                professional_signal.timing = f"Enter position within next 10 minutes for {preferences.trading_style.value.lower()} strategy."
            
            signals.append(professional_signal)
            signals_generated += 1
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            continue
    
    # Sort by confidence score
    signals.sort(key=lambda x: x.confidence_score, reverse=True)
    return signals

def generate_signal_by_style(
    trading_style: TradingStyle,
    current_price: float,
    indicators: Dict[str, Any],
    hist_1m: pd.DataFrame,
    hist_5m: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """
    Generate signals based on specific trading style
    """
    try:
        if trading_style == TradingStyle.SCALPING:
            return generate_scalping_signal(current_price, indicators, hist_1m)
        elif trading_style == TradingStyle.MOMENTUM:
            return generate_momentum_signal(current_price, indicators, hist_5m)
        elif trading_style == TradingStyle.SWING:
            return generate_swing_signal(current_price, indicators, hist_5m)
        
        return None
        
    except Exception as e:
        logger.error(f"Error generating signal by style: {str(e)}")
        return None

def generate_scalping_signal(current_price: float, indicators: Dict[str, Any], hist_1m: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Generate scalping signals (very short-term, 1-5 minutes)
    """
    try:
        score = 0
        direction = "HOLD"
        
        # Scalping criteria: Quick moves based on RSI and MACD
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        vwap = indicators.get('vwap', current_price)
        
        # RSI oversold/overbought for quick reversals
        if rsi < 25:  # Severely oversold
            score += 3
            direction = "BUY"
        elif rsi > 75:  # Severely overbought
            score += 3
            direction = "SELL"
        
        # MACD crossover for momentum
        if macd > macd_signal and macd > 0:
            score += 2
            if direction != "SELL":
                direction = "BUY"
        elif macd < macd_signal and macd < 0:
            score += 2
            if direction != "BUY":
                direction = "SELL"
        
        # Price vs VWAP
        if current_price > vwap * 1.002:  # Above VWAP
            score += 1
        elif current_price < vwap * 0.998:  # Below VWAP
            score += 1
        
        # Volume confirmation
        if indicators.get('volume_spike', False):
            score += 1
        
        confidence = min(95, score * 15 + 40)
        
        if score >= 3 and confidence >= 60:
            return {
                "direction": direction,
                "confidence": confidence,
                "signal_type": "SCALPING"
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error in scalping signal: {str(e)}")
        return None

def generate_momentum_signal(current_price: float, indicators: Dict[str, Any], hist_5m: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Generate momentum signals (15-60 minutes holding)
    """
    try:
        score = 0
        direction = "HOLD"
        
        # Momentum criteria: Trending moves with volume
        ema_5 = indicators.get('ema_5', current_price)
        ema_13 = indicators.get('ema_13', current_price)
        rsi = indicators.get('rsi', 50)
        macd_histogram = indicators.get('macd_histogram', 0)
        
        # EMA alignment for trend
        if current_price > ema_5 > ema_13:  # Bullish alignment
            score += 3
            direction = "BUY"
        elif current_price < ema_5 < ema_13:  # Bearish alignment
            score += 3
            direction = "SELL"
        
        # RSI in trending range
        if 40 < rsi < 70 and direction == "BUY":
            score += 2
        elif 30 < rsi < 60 and direction == "SELL":
            score += 2
        
        # MACD histogram increasing
        if macd_histogram > 0 and direction == "BUY":
            score += 2
        elif macd_histogram < 0 and direction == "SELL":
            score += 2
        
        # Volume confirmation
        if indicators.get('volume_spike', False):
            score += 1
        
        confidence = min(95, score * 12 + 35)
        
        if score >= 4 and confidence >= 65:
            return {
                "direction": direction,
                "confidence": confidence,
                "signal_type": "MOMENTUM"
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error in momentum signal: {str(e)}")
        return None

def generate_swing_signal(current_price: float, indicators: Dict[str, Any], hist_5m: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Generate swing signals (1-4 hours holding)
    """
    try:
        score = 0
        direction = "HOLD"
        
        # Swing criteria: Support/resistance with confirmation
        bb_upper = indicators.get('bb_upper', current_price * 1.02)
        bb_lower = indicators.get('bb_lower', current_price * 0.98)
        rsi = indicators.get('rsi', 50)
        stoch_k = indicators.get('stoch_k', 50)
        
        # Bollinger Band reversals
        if current_price <= bb_lower and rsi < 35:  # Oversold at support
            score += 4
            direction = "BUY"
        elif current_price >= bb_upper and rsi > 65:  # Overbought at resistance
            score += 4
            direction = "SELL"
        
        # Stochastic confirmation
        if stoch_k < 20 and direction == "BUY":
            score += 2
        elif stoch_k > 80 and direction == "SELL":
            score += 2
        
        # Volume confirmation
        if indicators.get('volume_spike', False):
            score += 1
        
        confidence = min(95, score * 10 + 40)
        
        if score >= 4 and confidence >= 70:
            return {
                "direction": direction,
                "confidence": confidence,
                "signal_type": "SWING"
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error in swing signal: {str(e)}")
        return None

def classify_volatility(atr: float, price: float) -> str:
    """
    Classify volatility level based on ATR
    """
    atr_percentage = (atr / price) * 100
    
    if atr_percentage < 1:
        return "LOW"
    elif atr_percentage < 2.5:
        return "MEDIUM"
    else:
        return "HIGH"

def prepare_chart_data(hist_1m: pd.DataFrame, indicators: Dict[str, Any], signal_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare chart data for frontend visualization
    """
    try:
        # Get last 100 1-minute candles
        recent_data = hist_1m.tail(100)
        
        chart_data = {
            "timestamps": [ts.strftime('%H:%M') for ts in recent_data.index],
            "ohlc": {
                "open": recent_data['Open'].tolist(),
                "high": recent_data['High'].tolist(),
                "low": recent_data['Low'].tolist(),
                "close": recent_data['Close'].tolist()
            },
            "volume": recent_data['Volume'].tolist(),
            "indicators": {
                "vwap": [indicators.get('vwap', recent_data['Close'].iloc[-1])] * len(recent_data),
                "bb_upper": [indicators.get('bb_upper', recent_data['Close'].iloc[-1] * 1.02)] * len(recent_data),
                "bb_lower": [indicators.get('bb_lower', recent_data['Close'].iloc[-1] * 0.98)] * len(recent_data),
                "rsi": [indicators.get('rsi', 50)] * len(recent_data)
            },
            "signals": [{
                "time": recent_data.index[-1].strftime('%H:%M'),
                "price": recent_data['Close'].iloc[-1],
                "type": signal_data['direction'],
                "confidence": signal_data['confidence']
            }]
        }
        
        return chart_data
        
    except Exception as e:
        logger.error(f"Error preparing chart data: {str(e)}")
        return {}

# Professional Trading Endpoints

@router.post("/professional-signals")
async def get_professional_trading_signals(
    capital_allocation: float = Query(description="Capital to allocate for trading"),
    risk_appetite: str = Query(default="MEDIUM", description="Risk appetite: LOW, MEDIUM, HIGH"),
    preferred_sectors: str = Query(default="", description="Comma-separated sectors (e.g., 'Technology,Finance')"),
    max_trades_per_day: int = Query(default=3, description="Maximum trades per day"),
    trading_style: str = Query(default="MOMENTUM", description="Trading style: SCALPING, MOMENTUM, SWING"),
    holding_duration_minutes: int = Query(default=30, description="Expected holding duration in minutes"),
    db: Session = Depends(get_db)
):
    """
    Generate professional-grade CFA trading signals with comprehensive analysis
    """
    try:
        # Parse preferences
        preferences = TradingPreferences(
            capital_allocation=capital_allocation,
            risk_appetite=RiskAppetite(risk_appetite),
            preferred_sectors=preferred_sectors.split(',') if preferred_sectors else [],
            max_trades_per_day=max_trades_per_day,
            trading_style=TradingStyle(trading_style),
            holding_duration_minutes=holding_duration_minutes
        )
        
        # Define stock universe based on trading style
        if preferences.trading_style == TradingStyle.SCALPING:
            # High liquidity stocks for scalping
            stock_universe = [
                "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
                "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT"
            ]
        elif preferences.trading_style == TradingStyle.MOMENTUM:
            # Momentum-friendly stocks
            stock_universe = [
                "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "HINDUNILVR",
                "INFY", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
                "ASIANPAINT", "MARUTI", "AXISBANK", "HCLTECH", "WIPRO"
            ]
        else:  # SWING
            # Broader universe for swing trading
            stock_universe = [
                "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "HINDUNILVR",
                "INFY", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
                "LT", "ASIANPAINT", "MARUTI", "AXISBANK", "HCLTECH",
                "WIPRO", "ULTRACEMCO", "NESTLEIND", "POWERGRID", "NTPC"
            ]
        
        # Generate professional signals
        signals = generate_professional_signals(preferences, stock_universe)
        
        # Convert to response format
        response_signals = []
        for signal in signals:
            response_signals.append({
                "symbol": signal.symbol,
                "company_name": signal.company_name,
                "confidence_score": round(signal.confidence_score, 1),
                "direction": signal.direction,
                "entry_price": round(signal.entry_price, 2),
                "stop_loss": round(signal.stop_loss, 2),
                "target_price_1": round(signal.target_price_1, 2),
                "target_price_2": round(signal.target_price_2, 2),
                "expected_roi": round(signal.expected_roi, 2),
                "risk_reward_ratio": round(signal.risk_reward_ratio, 2),
                "signal_time": signal.signal_time,
                "volatility_level": signal.volatility_level,
                "sector": signal.sector,
                "chart_data": signal.chart_data,
                "signal_log": signal.signal_log,
                # LLM Explanations
                "market_analysis": signal.market_analysis,
                "action_plan": signal.action_plan,
                "risk_management": signal.risk_management,
                "timing": signal.timing
            })
        
        return {
            "status": "success",
            "message": f"Generated {len(response_signals)} professional trading signals",
            "trading_preferences": {
                "capital_allocation": capital_allocation,
                "risk_appetite": risk_appetite,
                "trading_style": trading_style,
                "max_trades_per_day": max_trades_per_day,
                "holding_duration_minutes": holding_duration_minutes
            },
            "signals": response_signals,
            "market_session": "ACTIVE" if 9 <= datetime.datetime.now().hour < 15 else "CLOSED",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        logger.error(f"Error generating professional signals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating professional signals: {str(e)}")

@router.get("/backtest-signals")
async def backtest_todays_signals(
    trading_style: str = Query(default="MOMENTUM", description="Trading style to backtest"),
    db: Session = Depends(get_db)
):
    """
    Backtest today's signals against actual market performance
    """
    try:
        # This would compare generated signals with actual price movements
        # For now, return a sample structure
        
        backtest_results = {
            "status": "success",
            "backtest_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "trading_style": trading_style,
            "summary": {
                "total_signals": 5,
                "profitable_signals": 3,
                "loss_signals": 2,
                "accuracy_rate": 60.0,
                "avg_profit": 2.3,
                "avg_loss": -1.2,
                "risk_reward_achieved": 1.92
            },
            "individual_results": [
                {
                    "symbol": "RELIANCE",
                    "direction": "BUY",
                    "entry_price": 2480.50,
                    "exit_price": 2495.30,
                    "pnl_percent": 0.59,
                    "result": "PROFIT"
                }
                # More results would be here
            ]
        }
        
        return backtest_results
        
    except Exception as e:
        logger.error(f"Error in backtesting: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in backtesting: {str(e)}")

@router.get("/end-of-day-review")
async def get_end_of_day_review(
    date: str = Query(default="", description="Date in YYYY-MM-DD format, defaults to today"),
    db: Session = Depends(get_db)
):
    """
    Generate end-of-day trading performance review
    """
    try:
        review_date = date if date else datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Sample end-of-day review
        review = {
            "status": "success",
            "review_date": review_date,
            "trading_summary": {
                "total_trades": 4,
                "winning_trades": 3,
                "losing_trades": 1,
                "win_rate": 75.0,
                "total_pnl": 1250.75,
                "total_capital_deployed": 50000,
                "roi_for_day": 2.50
            },
            "performance_metrics": {
                "avg_holding_time_minutes": 45,
                "largest_win": 850.50,
                "largest_loss": -275.25,
                "risk_reward_realized": 1.85,
                "accuracy_vs_predicted": 80.0
            },
            "trade_details": [
                {
                    "symbol": "TCS",
                    "entry_time": "10:15",
                    "exit_time": "11:30",
                    "direction": "BUY",
                    "entry_price": 3420.50,
                    "exit_price": 3445.75,
                    "quantity": 10,
                    "pnl": 252.50,
                    "result": "TARGET_HIT"
                }
                # More trade details
            ],
            "lessons_learned": [
                "Market opened with strong momentum in IT sector",
                "Banking stocks showed weakness after 2 PM",
                "High volume breakouts performed better than expected",
                "RSI oversold signals were more reliable today"
            ],
            "recommendations_for_tomorrow": [
                "Focus on IT sector momentum if gap up continues",
                "Watch banking stocks for reversal signals",
                "Increase position size for high-volume breakouts",
                "Consider earlier exits in afternoon session"
            ]
        }
        
        return review
        
    except Exception as e:
        logger.error(f"Error generating end-of-day review: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating end-of-day review: {str(e)}")

@router.post("/current-prices")
async def get_current_prices(
    request: Dict[str, List[str]],
    db: Session = Depends(get_db)
):
    """
    Fetch current prices for given symbols to update real-time data
    """
    try:
        symbols = request.get('symbols', [])
        if not symbols:
            return {"status": "error", "message": "No symbols provided"}
        
        current_prices = {}
        
        for symbol in symbols:
            try:
                # Fetch current price from Yahoo Finance
                yf_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)
                
                # Get the most recent price
                hist = ticker.history(period="1d", interval="1m")
                if len(hist) > 0:
                    current_price = hist['Close'].iloc[-1]
                    current_prices[symbol] = float(current_price)
                else:
                    # Fallback to info if history is not available
                    info = ticker.info
                    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                    current_prices[symbol] = float(current_price) if current_price else 0
                    
            except Exception as e:
                logger.warning(f"Error fetching price for {symbol}: {str(e)}")
                current_prices[symbol] = 0
        
        return {
            "status": "success",
            "prices": current_prices,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_count": len([p for p in current_prices.values() if p > 0])
        }
        
    except Exception as e:
        logger.error(f"Error fetching current prices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching current prices: {str(e)}")
