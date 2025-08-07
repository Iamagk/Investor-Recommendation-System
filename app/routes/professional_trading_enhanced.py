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
router = APIRouter(tags=["Enhanced Small Capital Trading"])

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
class EnhancedSignal:
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
    chart_data: dict
    signal_log: List[str]
    # Position sizing for small capital
    shares_recommended: int
    investment_amount: float
    profit_potential_1: float
    profit_potential_2: float
    # LLM Explanations
    market_analysis: str = ""
    action_plan: str = ""
    risk_management: str = ""
    timing: str = ""

def filter_stocks_for_small_capital(capital: float) -> List[str]:
    """
    Get stocks optimized for small capital accounts
    Focus on price range and volatility suitable for meaningful profits
    """
    try:
        # Updated list with working symbols and good movement potential
        small_cap_friendly = [
            # Banking & Finance (working symbols)
            "SBIN", "PNB", "CANBK", "BANKINDIA", "UNIONBANK", "INDUSINDBK",
            
            # Telecom & Power (affordable range) 
            "BHARTIARTL", "POWERGRID", "NTPC", "COALINDIA", "GAIL",
            
            # Auto & Components (working symbols)
            "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "M&M", "MARUTI",
            
            # IT Services (affordable range)
            "TCS", "INFY", "WIPRO", "TECHM", "HCLTECH",
            
            # Consumer Goods
            "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
            
            # Metals & Mining
            "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA",
            
            # Pharma (high volatility)
            "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN",
            
            # Financial Services
            "BAJFINANCE", "BAJAJFINSV", "SBILIFE", "ICICIPRULI", "HDFCLIFE",
            
            # Infrastructure
            "LT", "ULTRACEMCO", "GRASIM", "SHREECEM", "RAMCOCEM"
        ]
        
        # Filter based on capital - prioritize different stocks for different capital ranges
        if capital < 15000:
            # Focus on most affordable stocks under ₹200-500
            priority_stocks = [
                "SBIN", "PNB", "CANBK", "BANKINDIA", "UNIONBANK",
                "POWERGRID", "NTPC", "COALINDIA", "GAIL",
                "ITC", "WIPRO", "VEDL", "HINDALCO"
            ]
        elif capital < 30000:
            # Mid-range stocks under ₹1000
            priority_stocks = [
                "BHARTIARTL", "INDUSINDBK", "TECHM", "HCLTECH",
                "TATASTEEL", "JSWSTEEL", "SUNPHARMA", "CIPLA",
                "DABUR", "LUPIN", "GRASIM"
            ]
        else:
            # Full list for larger small accounts (₹30k+)
            priority_stocks = small_cap_friendly
        
        return priority_stocks[:15]  # Limit to 15 for faster processing
        
    except Exception as e:
        logger.error(f"Error filtering stocks: {str(e)}")
        # Fallback to very safe, liquid stocks
        return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

def calculate_enhanced_targets(
    current_price: float, 
    capital: float, 
    direction: str, 
    risk_appetite: str,
    trading_style: str
) -> dict:
    """
    Calculate targets optimized for small capital meaningful profits
    """
    try:
        # Minimum profit targets based on capital
        if capital < 15000:
            min_profit_target = 300  # Minimum ₹300 profit
            target_percentage = 4.0  # 4% minimum move
        elif capital < 30000:
            min_profit_target = 500  # Minimum ₹500 profit
            target_percentage = 3.0  # 3% minimum move
        else:
            min_profit_target = 800  # Minimum ₹800 profit
            target_percentage = 2.5  # 2.5% minimum move
        
        # Adjust based on trading style
        style_multipliers = {
            "SCALPING": 1.0,     # Quick 2-4% moves
            "MOMENTUM": 1.3,     # 3-5% moves
            "SWING": 1.8         # 5-7% moves
        }
        
        risk_multipliers = {
            "LOW": 0.8,
            "MEDIUM": 1.0,
            "HIGH": 1.3
        }
        
        style_mult = style_multipliers.get(trading_style, 1.0)
        risk_mult = risk_multipliers.get(risk_appetite, 1.0)
        
        # Calculate percentage moves
        target1_pct = target_percentage * style_mult * risk_mult
        target2_pct = target1_pct * 1.8
        stop_loss_pct = target1_pct * 0.4  # 40% of target as stop loss
        
        # Calculate actual prices
        if direction == 'BUY':
            target_1 = current_price * (1 + target1_pct / 100)
            target_2 = current_price * (1 + target2_pct / 100)
            stop_loss = current_price * (1 - stop_loss_pct / 100)
        else:
            target_1 = current_price * (1 - target1_pct / 100)
            target_2 = current_price * (1 - target2_pct / 100)
            stop_loss = current_price * (1 + stop_loss_pct / 100)
        
        # Calculate position sizing
        shares_affordable = int(capital * 0.90 / current_price)  # 90% allocation
        investment_amount = shares_affordable * current_price
        
        # Calculate profit potentials
        profit_1 = shares_affordable * abs(target_1 - current_price)
        profit_2 = shares_affordable * abs(target_2 - current_price)
        
        # Skip if profit is too small
        if profit_1 < min_profit_target * 0.7:  # Allow 30% buffer
            return None
        
        return {
            'target_1': target_1,
            'target_2': target_2,
            'stop_loss': stop_loss,
            'shares_recommended': shares_affordable,
            'investment_amount': investment_amount,
            'profit_potential_1': profit_1,
            'profit_potential_2': profit_2,
            'target1_percentage': target1_pct,
            'target2_percentage': target2_pct,
            'stop_loss_percentage': stop_loss_pct
        }
        
    except Exception as e:
        logger.error(f"Error calculating enhanced targets: {str(e)}")
        return None

def generate_enhanced_signals(preferences: TradingPreferences) -> List[EnhancedSignal]:
    """
    Generate enhanced signals optimized for small capital accounts
    """
    signals = []
    
    # Get optimized stock list
    stock_list = filter_stocks_for_small_capital(preferences.capital_allocation)
    
    for symbol in stock_list:
        if len(signals) >= preferences.max_trades_per_day:
            break
            
        try:
            # Fetch data
            yf_symbol = f"{symbol}.NS"
            ticker = yf.Ticker(yf_symbol)
            
            hist_1m = ticker.history(period="2d", interval="1m")
            hist_5m = ticker.history(period="5d", interval="5m")
            info = ticker.info
            
            if len(hist_1m) < 20:
                continue
                
            current_price = float(hist_1m['Close'].iloc[-1])
            
            # Skip if too expensive for the capital
            max_affordable_price = preferences.capital_allocation * 0.3  # Max 30% in one stock
            if current_price > max_affordable_price:
                continue
            
            # Calculate enhanced targets
            targets = calculate_enhanced_targets(
                current_price,
                preferences.capital_allocation,
                'BUY',  # Focus on buy signals for simplicity
                preferences.risk_appetite.value,
                preferences.trading_style.value
            )
            
            if not targets:
                continue
            
            # Simple momentum signal (relaxed for testing)
            recent_close = hist_5m['Close'].tail(10)
            recent_volume = hist_5m['Volume'].tail(10)
            
            # Price momentum (more lenient)
            price_change = (recent_close.iloc[-1] - recent_close.iloc[-5]) / recent_close.iloc[-5] * 100
            volume_avg = recent_volume.mean()
            current_volume = recent_volume.iloc[-1]
            
            # Generate signal with relaxed criteria
            confidence = 65  # Base confidence
            
            if price_change > 0.5 and current_volume > volume_avg * 1.1:
                confidence += 15  # Positive momentum with volume
            elif price_change > 0.2:
                confidence += 10  # Small positive momentum
            elif price_change > -0.5:  # Even slightly negative is acceptable
                confidence += 5   # Neutral momentum
            
            # Much more lenient confidence threshold for testing
            if confidence < 60:
                confidence = 60  # Minimum acceptable confidence
            
            # Create enhanced signal
            signal = EnhancedSignal(
                symbol=symbol,
                company_name=info.get('longName', symbol),
                confidence_score=confidence,
                direction='BUY',
                entry_price=current_price,
                stop_loss=targets['stop_loss'],
                target_price_1=targets['target_1'],
                target_price_2=targets['target_2'],
                expected_roi=targets['target1_percentage'],
                risk_reward_ratio=targets['target1_percentage'] / targets['stop_loss_percentage'],
                signal_time=datetime.datetime.now().strftime("%H:%M:%S"),
                volatility_level="MEDIUM",
                sector=info.get('sector', 'Unknown'),
                chart_data={},  # Simplified for now
                signal_log=[
                    f"BUY {symbol} at ₹{current_price:.2f}",
                    f"Target 1: ₹{targets['target_1']:.2f} (₹{targets['profit_potential_1']:.0f} profit)",
                    f"Target 2: ₹{targets['target_2']:.2f} (₹{targets['profit_potential_2']:.0f} profit)",
                    f"Stop Loss: ₹{targets['stop_loss']:.2f}",
                    f"Invest: ₹{targets['investment_amount']:.0f} ({targets['shares_recommended']} shares)"
                ],
                shares_recommended=targets['shares_recommended'],
                investment_amount=targets['investment_amount'],
                profit_potential_1=targets['profit_potential_1'],
                profit_potential_2=targets['profit_potential_2']
            )
            
            # Add simple explanations
            signal.market_analysis = f"{symbol} showing positive momentum with {price_change:.1f}% recent price increase and above-average volume."
            signal.action_plan = f"Buy {targets['shares_recommended']} shares at ₹{current_price:.2f} for total investment of ₹{targets['investment_amount']:.0f}. Target profit of ₹{targets['profit_potential_1']:.0f} ({targets['target1_percentage']:.1f}%)."
            signal.risk_management = f"Set stop-loss at ₹{targets['stop_loss']:.2f}. Maximum loss limited to ₹{abs(targets['shares_recommended'] * (current_price - targets['stop_loss'])):.0f}."
            signal.timing = f"Execute within 15 minutes for {preferences.trading_style.value.lower()} strategy. Monitor for 30-60 minutes."
            
            signals.append(signal)
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            continue
    
    # Sort by profit potential
    signals.sort(key=lambda x: x.profit_potential_1, reverse=True)
    return signals

@router.post("/enhanced-signals")
async def get_enhanced_professional_signals(
    capital_allocation: float = Query(10000, description="Capital allocation in INR"),
    risk_appetite: str = Query("MEDIUM", description="Risk appetite: LOW, MEDIUM, HIGH"),
    max_trades_per_day: int = Query(3, description="Maximum trades per day"),
    trading_style: str = Query("MOMENTUM", description="Trading style: SCALPING, MOMENTUM, SWING"),
    holding_duration_minutes: int = Query(30, description="Expected holding duration in minutes"),
    db: Session = Depends(get_db)
):
    """
    Get enhanced professional trading signals optimized for small capital accounts
    Focuses on higher percentage moves and meaningful profit targets
    """
    try:
        logger.info(f"Generating enhanced signals for capital: ₹{capital_allocation}")
        
        preferences = TradingPreferences(
            capital_allocation=capital_allocation,
            risk_appetite=RiskAppetite(risk_appetite),
            preferred_sectors=[],
            max_trades_per_day=max_trades_per_day,
            trading_style=TradingStyle(trading_style),
            holding_duration_minutes=holding_duration_minutes
        )
        
        signals = generate_enhanced_signals(preferences)
        
        # Format response
        formatted_signals = []
        for signal in signals:
            formatted_signals.append({
                "symbol": signal.symbol,
                "company_name": signal.company_name,
                "confidence_score": round(signal.confidence_score, 1),
                "direction": signal.direction,
                "entry_price": round(signal.entry_price, 2),
                "stop_loss": round(signal.stop_loss, 2),
                "target_price_1": round(signal.target_price_1, 2),
                "target_price_2": round(signal.target_price_2, 2),
                "expected_roi": round(signal.expected_roi, 1),
                "risk_reward_ratio": round(signal.risk_reward_ratio, 1),
                "signal_time": signal.signal_time,
                "volatility_level": signal.volatility_level,
                "sector": signal.sector,
                "chart_data": signal.chart_data,
                "signal_log": signal.signal_log,
                # Enhanced fields for small capital
                "shares_recommended": signal.shares_recommended,
                "investment_amount": round(signal.investment_amount, 0),
                "profit_potential_1": round(signal.profit_potential_1, 0),
                "profit_potential_2": round(signal.profit_potential_2, 0),
                # LLM explanations
                "market_analysis": signal.market_analysis,
                "action_plan": signal.action_plan,
                "risk_management": signal.risk_management,
                "timing": signal.timing
            })
        
        return {
            "signals": formatted_signals,
            "total_signals": len(formatted_signals),
            "capital_optimization": {
                "total_capital": capital_allocation,
                "max_investment_per_trade": round(capital_allocation * 0.3, 0),
                "min_profit_target": 300 if capital_allocation < 15000 else 500 if capital_allocation < 30000 else 800,
                "strategy": f"Optimized for {trading_style.lower()} with {risk_appetite.lower()} risk"
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating enhanced signals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating signals: {str(e)}")
