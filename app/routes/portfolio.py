# app/routes/portfolio.py

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import datetime

from app.db import get_db
from app.ml.predictor import recommend_stocks, recommend_mutual_funds, recommend_gold
from app.recommender.recommender import recommend_assets

router = APIRouter()

class InvestmentRequest(BaseModel):
    amount: float
    risk_tolerance: int  # 1-10 scale
    expected_return: int  # percentage
    selected_options: List[str]  # ['stocks', 'mutualFunds', 'gold']

class InvestmentRecommendation(BaseModel):
    recommended_allocation: Dict[str, int]
    expected_return: float
    risk_score: int
    recommendations: List[str]
    detailed_investments: Dict[str, Any]
    
    class Config:
        arbitrary_types_allowed = True

def map_risk_profile(risk_tolerance: int) -> str:
    """Map numeric risk tolerance to profile string"""
    if risk_tolerance <= 3:
        return "conservative"
    elif risk_tolerance <= 7:
        return "balanced"
    else:
        return "aggressive"

def calculate_asset_allocation(risk_tolerance: int, expected_return: int, selected_options: List[str]) -> Dict[str, int]:
    """Calculate optimal asset allocation based on user preferences"""
    
    # Base allocations by risk profile
    if risk_tolerance <= 3:  # Conservative
        base_stocks = 20
        base_mutual_funds = 60
        base_gold = 20
    elif risk_tolerance <= 7:  # Balanced
        base_stocks = 50
        base_mutual_funds = 35
        base_gold = 15
    else:  # Aggressive
        base_stocks = 70
        base_mutual_funds = 25
        base_gold = 5
    
    # Adjust based on expected return
    if expected_return > 15:  # High return expectation
        base_stocks = min(80, base_stocks + 20)
        base_mutual_funds = max(10, base_mutual_funds - 10)
        base_gold = max(5, base_gold - 10)
    elif expected_return < 8:  # Low return expectation
        base_stocks = max(10, base_stocks - 15)
        base_mutual_funds = min(60, base_mutual_funds + 10)
        base_gold = min(30, base_gold + 5)
    
    # Filter based on selected options and normalize
    allocation = {}
    if 'stocks' in selected_options:
        allocation['stocks'] = base_stocks
    else:
        allocation['stocks'] = 0
        
    if 'mutualFunds' in selected_options:
        allocation['mutualFunds'] = base_mutual_funds
    else:
        allocation['mutualFunds'] = 0
        
    if 'gold' in selected_options:
        allocation['gold'] = base_gold
    else:
        allocation['gold'] = 0
    
    # Normalize to 100%
    total = sum(allocation.values())
    if total > 0:
        allocation = {k: round((v / total) * 100) for k, v in allocation.items()}
    
    return allocation

@router.post("/predict", response_model=InvestmentRecommendation)
def get_investment_prediction(
    request: InvestmentRequest,
    db: Session = Depends(get_db)
):
    """
    Get AI-powered investment recommendations based on user preferences
    """
    try:
        # Get comprehensive recommendations using our enhanced function
        comprehensive_result = recommend_assets(db, top_n=5, use_ml=True, use_realtime=False)
        
        # Calculate optimal allocation
        allocation = calculate_asset_allocation(
            request.risk_tolerance, 
            request.expected_return, 
            request.selected_options
        )
        
        # Initialize detailed investments
        detailed_investments = {}
        recommendations = []
        
        # Process stocks
        if 'stocks' in request.selected_options and allocation['stocks'] > 0:
            stocks_data = comprehensive_result.get('recommendations', {}).get('stocks', [])
            if stocks_data:
                # Calculate average return from sectors
                avg_return = sum(sector.get('predicted_return', 0) for sector in stocks_data) / len(stocks_data)
                
                # Get top stock picks from sectors
                top_picks = []
                for sector in stocks_data[:3]:  # Top 3 sectors
                    sector_stocks = sector.get('stocks', [])
                    if sector_stocks:
                        stock = sector_stocks[0]  # Get top stock from sector
                        top_picks.append({
                            'name': stock.get('company_name', stock.get('symbol', 'Unknown')),
                            'symbol': stock.get('symbol', 'N/A'),
                            'sector': sector.get('sector', 'Unknown'),
                            'predicted_return': sector.get('predicted_return', 0),
                            'current_performance': stock.get('current_performance', 0),
                            'timing_analysis': stock.get('timing_analysis', {}),
                            'investment_strategy': stock.get('investment_strategy', ''),
                            'commentary': stock.get('commentary', '')
                        })
                
                detailed_investments['stocks'] = {
                    'allocation_percent': allocation['stocks'],
                    'allocation_amount': (allocation['stocks'] / 100) * request.amount,
                    'top_picks': top_picks,
                    'average_expected_return': avg_return,
                    'comprehensive_data': stocks_data  # Include full comprehensive data
                }
                recommendations.append(f"Stock portfolio diversified across {len(stocks_data)} high-potential sectors")
        
        # Process mutual funds
        if 'mutualFunds' in request.selected_options and allocation['mutualFunds'] > 0:
            mf_data = comprehensive_result.get('recommendations', {}).get('mutual_funds', [])
            if mf_data:
                # Calculate average return from sectors
                avg_return = sum(sector.get('predicted_return', 0) for sector in mf_data) / len(mf_data)
                
                # Get top fund picks from sectors
                top_picks = []
                for sector in mf_data[:3]:  # Top 3 sectors
                    sector_funds = sector.get('funds', [])
                    if sector_funds:
                        fund = sector_funds[0]  # Get top fund from sector
                        top_picks.append({
                            'name': fund.get('fund_name', 'Unknown Fund'),
                            'full_name': fund.get('full_name', ''),
                            'sector': sector.get('sector', 'Unknown'),
                            'predicted_return': sector.get('predicted_return', 0),
                            'current_performance': fund.get('current_performance', 0),
                            'commentary': fund.get('commentary', ''),
                            'investment_strategy': fund.get('investment_strategy', '')
                        })
                
                detailed_investments['mutualFunds'] = {
                    'allocation_percent': allocation['mutualFunds'],
                    'allocation_amount': (allocation['mutualFunds'] / 100) * request.amount,
                    'top_picks': top_picks,
                    'average_expected_return': avg_return,
                    'comprehensive_data': mf_data  # Include full comprehensive data
                }
                recommendations.append(f"Mutual fund portfolio spanning {len(mf_data)} diversified sectors")
        
        # Process gold
        if 'gold' in request.selected_options and allocation['gold'] > 0:
            gold_data = comprehensive_result.get('recommendations', {}).get('gold', [])
            if gold_data:
                # Calculate average return
                avg_return = sum(sector.get('predicted_return', 0) for sector in gold_data) / len(gold_data)
                
                # Get top gold investment options
                top_picks = []
                for sector in gold_data[:3]:  # Top 3 options
                    gold_investments = sector.get('gold_investments', [])
                    if gold_investments:
                        investment = gold_investments[0]
                        top_picks.append({
                            'type': investment.get('investment_type', 'Gold ETF'),
                            'sector': sector.get('sector', 'Precious Metals'),
                            'predicted_return': sector.get('predicted_return', 0),
                            'commentary': investment.get('commentary', ''),
                            'investment_strategy': investment.get('investment_strategy', '')
                        })
                
                detailed_investments['gold'] = {
                    'allocation_percent': allocation['gold'],
                    'allocation_amount': (allocation['gold'] / 100) * request.amount,
                    'top_picks': top_picks,
                    'average_expected_return': avg_return,
                    'comprehensive_data': gold_data  # Include full comprehensive data
                }
                recommendations.append(f"Gold investments for portfolio stability and inflation hedge")
        
        # Calculate overall expected return
        overall_expected_return = 0
        total_weight = 0
        for asset_type, details in detailed_investments.items():
            weight = details['allocation_percent'] / 100
            return_rate = details['average_expected_return']
            overall_expected_return += weight * return_rate
            total_weight += weight
        
        if total_weight > 0:
            overall_expected_return = overall_expected_return / total_weight
        else:
            overall_expected_return = request.expected_return
        
        # Add general recommendations
        risk_profile = map_risk_profile(request.risk_tolerance)
        recommendations.extend([
            f"Based on your {risk_profile} risk profile, this allocation balances growth and stability",
            f"Your ${request.amount:,.0f} investment is diversified across {len(request.selected_options)} asset classes",
            "Consider rebalancing quarterly to maintain optimal allocation",
            f"Expected portfolio return: {overall_expected_return:.1f}% annually"
        ])
        
        return InvestmentRecommendation(
            recommended_allocation=allocation,
            expected_return=round(overall_expected_return, 1),
            risk_score=request.risk_tolerance,
            recommendations=recommendations,
            detailed_investments=detailed_investments
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating investment recommendations: {str(e)}"
        )

@router.get("/health")
def portfolio_health_check():
    """Health check for portfolio service"""
    return {
        "status": "healthy",
        "service": "portfolio_recommendations",
        "timestamp": datetime.datetime.now().isoformat()
    }
