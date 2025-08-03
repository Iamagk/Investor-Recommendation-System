from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
import requests
import json
import io
from contextlib import redirect_stdout
from datetime import datetime
from app.db import get_db

router = APIRouter(tags=["Backtest"])

class BacktestRequest(BaseModel):
    budget: float
    risk_profile: str = "moderate"
    investment_horizon: str = "long_term"
    top_n: int = 3

def generate_fresh_comprehensive_analysis(budget: float, risk_profile: str, investment_horizon: str, top_n: int = 3):
    """Generate fresh comprehensive analysis with the same format as run_backtest.py"""
    
    # Get fresh ML recommendations
    asset_types = ['stocks', 'mutual-funds', 'gold']
    all_recommendations = {}
    
    output_lines = []
    output_lines.append("Fetching fresh ML recommendations for comprehensive analysis...")
    
    for asset_type in asset_types:
        output_lines.append(f"Getting {asset_type} recommendations...")
        try:
            response = requests.get(f"http://localhost:8000/ml/recommend/{asset_type}?top_n={top_n}")
            if response.status_code == 200:
                all_recommendations[asset_type] = response.json()
            else:
                output_lines.append(f"Error getting {asset_type} recommendations: {response.status_code}")
        except Exception as e:
            output_lines.append(f"Error in {asset_type} recommendations: {str(e)}")
    
    if not all_recommendations:
        output_lines.append("Could not fetch any recommendations for analysis")
        return "\n".join(output_lines)
    
    output_lines.append(f"\n" + "="*60)
    output_lines.append(f"COMPREHENSIVE INVESTMENT ANALYSIS RESULTS")
    output_lines.append(f"="*60)
    output_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    output_lines.append(f"Strategy: ML-based Multi-Asset Recommendations")
    output_lines.append(f"Budget: ₹{budget:,.0f}")
    output_lines.append(f"Risk Profile: {risk_profile.title()}")
    output_lines.append(f"Investment Horizon: {investment_horizon.replace('_', ' ').title()}")
    output_lines.append(f"Assets Covered: {', '.join(all_recommendations.keys())}")
    
    # Import timing analysis function
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backtest'))
    from app.backtest.run_backtest import get_stock_timing_analysis
    
    total_portfolio_score = 0
    total_predicted_return = 0
    total_asset_count = 0
    
    # Process each asset type with detailed analysis
    for asset_type, recommendations in all_recommendations.items():
        output_lines.append(f"\n" + "="*50)
        output_lines.append(f"📊 {asset_type.upper().replace('-', ' ')} RECOMMENDATIONS")
        output_lines.append(f"="*50)
        output_lines.append(f"Method: {recommendations.get('method', 'ML-based predictions with weighted scoring')}")
        
        asset_total_return = 0
        asset_count = 0
        
        for i, rec in enumerate(recommendations.get('recommendations', []), 1):
            sector = rec.get('sector', 'Unknown')
            predicted_return = rec.get('predicted_return', 0)
            score = rec.get('score', 0)
            investment_count = rec.get('investment_count', 0)
            avg_price = rec.get('avg_price', 0)
            top_performers = rec.get('top_performers', '[]')
            
            output_lines.append(f"\n{i}. Sector: {sector}")
            output_lines.append(f"   Predicted Return: {predicted_return:.2f}%")
            output_lines.append(f"   ML Score: {score:.2f}")
            output_lines.append(f"   Investment Count: {investment_count}")
            output_lines.append(f"   Avg Price: ₹{avg_price:.2f}")
            
            # Add sector-level conversational explanation
            if predicted_return > 6:
                sector_outlook = "shows exceptional growth potential"
            elif predicted_return > 4:
                sector_outlook = "demonstrates strong growth prospects"
            elif predicted_return > 2:
                sector_outlook = "offers steady growth opportunities"
            else:
                sector_outlook = "provides stability with moderate returns"
            
            if score > 6:
                ml_confidence = "high confidence"
            elif score > 4:
                ml_confidence = "good confidence"
            else:
                ml_confidence = "moderate confidence"
            
            output_lines.append(f"   💬 SECTOR ANALYSIS: The {sector} sector {sector_outlook} with our ML model showing {ml_confidence} (score: {score:.2f}).")
            output_lines.append(f"   💬 With {investment_count} investment opportunities available, this sector offers {'good diversification' if investment_count > 3 else 'focused investment options'}.")
            
            # Parse and display specific investments with detailed analysis
            try:
                if isinstance(top_performers, str):
                    investments = json.loads(top_performers.replace("'", '"'))
                else:
                    investments = top_performers
                
                if investments:
                    if asset_type == 'stocks':
                        output_lines.append(f"   📈 RECOMMENDED STOCKS TO BUY:")
                        for j, investment in enumerate(investments, 1):
                            symbol = investment.get('symbol', 'N/A')
                            name = investment.get('name', 'N/A')
                            change_percent = investment.get('change_percent', 0)
                            output_lines.append(f"      {j}. {symbol} - {name}")
                            output_lines.append(f"         Current Performance: {change_percent:.2f}%")
                            
                            # Add conversational explanation
                            if change_percent > 5:
                                performance_desc = "showing strong momentum"
                            elif change_percent > 2:
                                performance_desc = "performing well"
                            elif change_percent > 0:
                                performance_desc = "showing positive growth"
                            else:
                                performance_desc = "currently facing challenges but has potential"
                            
                            output_lines.append(f"         💬 {name} is {performance_desc} with {change_percent:.2f}% current performance.")
                            
                            # Get detailed timing analysis for stocks
                            timing_analysis = get_stock_timing_analysis(symbol)
                            if timing_analysis:
                                expected_return = timing_analysis['expected_return']
                                holding_days = timing_analysis['holding_period']
                                volatility = timing_analysis['volatility']
                                
                                # Risk assessment
                                if volatility < 4:
                                    risk_level = "low risk"
                                elif volatility < 6:
                                    risk_level = "moderate risk"
                                else:
                                    risk_level = "higher risk"
                                
                                # Return assessment
                                if expected_return > 12:
                                    return_desc = "excellent"
                                elif expected_return > 8:
                                    return_desc = "good"
                                else:
                                    return_desc = "steady"
                                
                                output_lines.append(f"         📝 INVESTMENT STRATEGY: Buy {symbol} on {timing_analysis['entry_date']} at ₹{timing_analysis['entry_price']:.2f}.")
                                output_lines.append(f"         📝 This is a {risk_level} investment with {return_desc} potential returns of {expected_return:.2f}% over {holding_days} days.")
                                output_lines.append(f"         📝 Set your stop loss at ₹{timing_analysis['stop_loss']:.2f} to limit losses, and target ₹{timing_analysis['target_price']:.2f} for profit booking.")
                                
                                output_lines.append(f"         ⏰ TIMING ANALYSIS:")
                                output_lines.append(f"            📅 Entry Date: {timing_analysis['entry_date']}")
                                output_lines.append(f"            💰 Entry Price: ₹{timing_analysis['entry_price']:.2f}")
                                output_lines.append(f"            📅 Exit Date: {timing_analysis['exit_date']}")
                                output_lines.append(f"            💰 Exit Price: ₹{timing_analysis['exit_price']:.2f}")
                                output_lines.append(f"            📊 Expected Return: {timing_analysis['expected_return']:.2f}%")
                                output_lines.append(f"            📈 Support Level: ₹{timing_analysis['support_level']:.2f}")
                                output_lines.append(f"            📉 Resistance Level: ₹{timing_analysis['resistance_level']:.2f}")
                                output_lines.append(f"            🛑 Stop Loss: ₹{timing_analysis['stop_loss']:.2f}")
                                output_lines.append(f"            🎯 Target Price: ₹{timing_analysis['target_price']:.2f}")
                                output_lines.append(f"            ⏳ Holding Period: {timing_analysis['holding_period']} days")
                                output_lines.append(f"            📊 Volatility: {timing_analysis['volatility']:.2f}%")
                    elif asset_type == 'mutual-funds':
                        output_lines.append(f"   💼 RECOMMENDED FUNDS TO BUY:")
                        for j, investment in enumerate(investments, 1):
                            symbol = investment.get('symbol', 'N/A')
                            name = investment.get('name', 'N/A')
                            change_percent = investment.get('change_percent', 0)
                            output_lines.append(f"      {j}. {symbol} - {name}")
                            output_lines.append(f"         Current Performance: {change_percent:.2f}%")
                            
                            # Add conversational explanation for mutual funds
                            if change_percent > 3:
                                fund_desc = "This fund is performing exceptionally well"
                            elif change_percent > 1:
                                fund_desc = "This fund shows solid performance"
                            elif change_percent > 0:
                                fund_desc = "This fund is maintaining positive growth"
                            else:
                                fund_desc = "This fund offers good long-term potential despite current challenges"
                            
                            output_lines.append(f"         💬 {fund_desc} and is suitable for investors looking for {sector.lower()} sector exposure.")
                            output_lines.append(f"         📝 INVESTMENT STRATEGY: Consider systematic investment (SIP) for rupee cost averaging in this {sector} focused fund.")
                            
                    else:  # gold
                        output_lines.append(f"   🥇 RECOMMENDED GOLD INVESTMENTS:")
                        for j, investment in enumerate(investments, 1):
                            name = investment if isinstance(investment, str) else investment.get('name', 'Gold')
                            output_lines.append(f"      {j}. {name}")
                            output_lines.append(f"         💬 Gold serves as a hedge against inflation and market volatility.")
                            output_lines.append(f"         📝 INVESTMENT STRATEGY: Allocate 10-15% of your portfolio to gold for diversification and stability during uncertain times.")
                else:
                    output_lines.append(f"   📈 RECOMMENDED INVESTMENTS: Data not available")
            except Exception as e:
                output_lines.append(f"   📈 RECOMMENDED INVESTMENTS: Could not parse investment data")
            
            asset_total_return += predicted_return
            asset_count += 1
            total_portfolio_score += score
        
        if asset_count > 0:
            avg_asset_return = asset_total_return / asset_count
            output_lines.append(f"\n📈 {asset_type.upper()} SUMMARY:")
            output_lines.append(f"   Average Predicted Return: {avg_asset_return:.2f}%")
            output_lines.append(f"   Total Score: {sum(rec.get('score', 0) for rec in recommendations.get('recommendations', [])):.2f}")
            output_lines.append(f"   Diversification: {len(set(rec.get('sector') for rec in recommendations.get('recommendations', [])))} sectors")
            
            total_predicted_return += asset_total_return
            total_asset_count += asset_count
    
    # Calculate comprehensive portfolio metrics
    if total_asset_count > 0:
        overall_avg_return = total_predicted_return / total_asset_count
        
        output_lines.append(f"\n" + "="*60)
        output_lines.append(f"🏆 OVERALL PORTFOLIO ANALYSIS")
        output_lines.append(f"="*60)
        output_lines.append(f"💰 Budget Allocation: ₹{budget:,.0f}")
        output_lines.append(f"📊 Overall Avg Predicted Return: {overall_avg_return:.2f}%")
        output_lines.append(f"📈 Total Portfolio Score: {total_portfolio_score:.2f}")
        output_lines.append(f"🎯 Asset Class Diversification: {len(all_recommendations)} types")
        output_lines.append(f"💡 Total Investment Opportunities: {sum(sum(rec.get('investment_count', 0) for rec in recs.get('recommendations', [])) for recs in all_recommendations.values())}")
        
        # Add asset-specific metrics
        for asset_type, recommendations in all_recommendations.items():
            asset_recs = recommendations.get('recommendations', [])
            if asset_recs:
                asset_avg_return = sum(rec.get('predicted_return', 0) for rec in asset_recs) / len(asset_recs)
                output_lines.append(f"📊 {asset_type.title()} Avg Return: {asset_avg_return:.2f}%")
    
    return "\n".join(output_lines)

@router.post("/comprehensive", summary="Get fresh comprehensive investment analysis")
def get_comprehensive_backtest(
    request: BacktestRequest,
    db: Session = Depends(get_db)
):
    """
    Returns fresh comprehensive investment analysis with detailed sector analysis,
    timing recommendations, and conversational explanations - generates new predictions each time.
    
    This endpoint provides rich, detailed analysis similar to run_backtest.py format
    but with fresh ML predictions based on current data.
    """
    try:
        # Generate fresh comprehensive analysis
        comprehensive_output = generate_fresh_comprehensive_analysis(
            budget=request.budget,
            risk_profile=request.risk_profile,
            investment_horizon=request.investment_horizon,
            top_n=request.top_n
        )
        
        # Calculate metrics from the fresh analysis
        metrics = {
            'budget': request.budget,
            'risk_profile': request.risk_profile,
            'investment_horizon': request.investment_horizon,
            'analysis_timestamp': datetime.now().isoformat(),
            'fresh_predictions': True
        }
        
        return {
            "status": "success",
            "message": "Fresh comprehensive investment analysis completed",
            "backtest_output": comprehensive_output,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "fresh_comprehensive_analysis"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating fresh comprehensive analysis: {str(e)}")
