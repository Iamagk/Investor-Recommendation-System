from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
import datetime
from app.db import get_db
from app.recommender.recommender import recommend_assets, recommend_stocks, recommend_mutual_funds, recommend_gold, recommend_stocks_from_dataframe
from app.utils.data_loader import load_stock_features

router = APIRouter(tags=["Recommendations"])

@router.get("/", summary="Get top investment recommendations")
def get_recommendations(
    db: Session = Depends(get_db),
    top_n: int = 5,
    use_ml: bool = True,
    use_realtime: bool = False
):
    """
    Returns top N recommendations from stocks, mutual funds, and gold sectors.
    
    Parameters:
    - top_n: Number of recommendations per asset type
    - use_ml: Enable ML-powered predictions for stocks (default: True)
    - use_realtime: Fetch real-time prices (default: False, slower but current)
    """
    result = recommend_assets(db, top_n=top_n, use_ml=use_ml, use_realtime=use_realtime)
    return result

@router.get("/stocks", summary="Get stock recommendations only")
def get_stock_recommendations(
    db: Session = Depends(get_db),
    top_n: int = 5,
    use_realtime: bool = False
):
    """
    Get top stock recommendations with ML predictions
    
    Parameters:
    - top_n: Number of stock recommendations
    - use_realtime: Fetch real-time prices (default: False)
    """
    try:
        stocks = recommend_stocks(db, top_n, use_realtime=use_realtime)
        return {
            "status": "success",
            "message": f"Top {top_n} ML-enhanced stock recommendations",
            "ml_enhanced": True,
            "realtime_prices": use_realtime,
            "stocks": stocks,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stock recommendations: {str(e)}")

@router.get("/stocks/dataframe", summary="Get stock recommendations using DataFrame loader")
def get_stock_recommendations_dataframe(use_realtime: bool = Query(False)):
    """
    Get stock recommendations using the DataFrame-based approach
    
    Parameters:
    - use_realtime: Fetch real-time prices (default: False)
    
    This endpoint loads stock features directly from the database into a DataFrame
    and uses the original recommend_stocks function format.
    """
    try:
        # Load stock features from database
        df = load_stock_features()
        
        if df.empty:
            return {
                "status": "warning",
                "message": "No stock data available in database",
                "recommendations": [],
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Get recommendations using DataFrame approach
        recommendations = recommend_stocks_from_dataframe(df, use_realtime=use_realtime)
        
        return {
            "status": "success",
            "message": f"Stock recommendations using DataFrame approach",
            "data_source": "database_dataframe",
            "stocks_analyzed": len(df),
            "ml_enhanced": True,
            "realtime_prices": use_realtime,
            "recommendations": recommendations,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting DataFrame-based stock recommendations: {str(e)}")

@router.get("/mutual-funds", summary="Get mutual fund recommendations only")
def get_mutual_fund_recommendations(
    db: Session = Depends(get_db),
    top_n: int = 5
):
    """Get top mutual fund recommendations based on analysis data"""
    try:
        mutual_funds = recommend_mutual_funds(db, top_n)
        return {
            "status": "success",
            "message": f"Top {top_n} mutual fund recommendations",
            "mutual_funds": mutual_funds,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting mutual fund recommendations: {str(e)}")

@router.get("/gold", summary="Get gold investment recommendation")
def get_gold_recommendation(db: Session = Depends(get_db)):
    """Get gold investment recommendation based on analysis data"""
    try:
        gold = recommend_gold(db)
        return {
            "status": "success",
            "message": "Gold investment recommendation",
            "gold": gold,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting gold recommendation: {str(e)}")

@router.get("/portfolio")
def get_portfolio_recommendations(
    db: Session = Depends(get_db),
    risk_level: str = "moderate",
    top_n: int = 3
):
    """Get personalized portfolio recommendations based on risk level"""
    risk_levels = ["conservative", "moderate", "aggressive"]
    
    if risk_level not in risk_levels:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid risk level. Choose from: {', '.join(risk_levels)}"
        )
    
    # Get recommendations from database
    recommendations = recommend_assets(db, top_n=top_n)
    
    # Adjust allocations based on risk level
    allocation = {
        "stocks": 60 if risk_level == "moderate" else (40 if risk_level == "conservative" else 80),
        "bonds": 30 if risk_level == "moderate" else (50 if risk_level == "conservative" else 15),
        "gold": 10 if risk_level == "moderate" else (10 if risk_level == "conservative" else 5)
    }
    
    return {
        "status": "success",
        "message": f"Portfolio recommendations for {risk_level} risk level",
        "risk_level": risk_level,
        "portfolio": {
            "allocation": allocation,
            "recommended_investments": recommendations.get("recommendations", {})
        },
        "timestamp": datetime.datetime.now().isoformat()
    }

@router.get("/sectors")
def get_sector_recommendations(db: Session = Depends(get_db)):
    """Get sector-wise investment recommendations based on real data"""
    try:
        # Get stock recommendations to analyze sectors
        stocks = recommend_stocks(db, top_n=10)
        
        # Group by sectors and calculate average scores
        sector_scores = {}
        for stock in stocks:
            sector = stock.get("sector", "Unknown")
            if sector not in sector_scores:
                sector_scores[sector] = {"scores": [], "count": 0}
            sector_scores[sector]["scores"].append(stock.get("score", 0))
            sector_scores[sector]["count"] += 1
        
        # Calculate average scores and recommendations
        sector_recommendations = {}
        for sector, data in sector_scores.items():
            avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
            recommendation = "buy" if avg_score > 7 else "hold" if avg_score > 5 else "sell"
            sector_recommendations[sector] = {
                "score": round(avg_score, 2),
                "recommendation": recommendation,
                "stock_count": data["count"]
            }
        
        return {
            "status": "success",
            "message": "Sector-wise investment recommendations based on real analysis",
            "sectors": sector_recommendations,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sector recommendations: {str(e)}")

@router.get("/portfolio", summary="Get optimized portfolio with budget allocation and dynamic weights")
def get_portfolio_optimization(
    db: Session = Depends(get_db),
    budget: float = Query(..., description="Investment budget in currency"),
    top_n: int = Query(5, description="Number of top recommendations"),
    use_ml: bool = Query(True, description="Use ML predictions"),
    risk_profile: str = Query("balanced", description="Risk profile: conservative, balanced, aggressive, momentum_focused, value_focused"),
    return_weight: float = Query(None, description="Custom return weight (0.0-1.0)"),
    volatility_weight: float = Query(None, description="Custom volatility weight (0.0-0.5)"),
    momentum_weight: float = Query(None, description="Custom momentum weight (0.0-1.0)")
):
    """
    Returns optimized portfolio allocation with dynamic weights based on market conditions and user preferences.
    
    Parameters:
    - budget: Total investment budget
    - top_n: Number of recommendations per asset type
    - use_ml: Enable ML-powered predictions (default: True)
    - risk_profile: Risk tolerance (conservative, balanced, aggressive, momentum_focused, value_focused)
    - return_weight: Custom weight for returns (optional, overrides profile defaults)
    - volatility_weight: Custom weight for volatility penalty (optional)
    - momentum_weight: Custom weight for momentum (optional)
    """
    from app.recommender.recommender import full_recommendation_with_budget
    
    # Prepare user preferences if custom weights are provided
    user_preferences = {}
    if return_weight is not None:
        user_preferences["return"] = return_weight - 0.4  # Adjustment from default
    if volatility_weight is not None:
        user_preferences["volatility"] = volatility_weight - 0.2  # Adjustment from default
    if momentum_weight is not None:
        user_preferences["momentum"] = momentum_weight - 0.4  # Adjustment from default
    
    result = full_recommendation_with_budget(
        db, 
        budget=budget, 
        top_n=top_n, 
        use_ml=use_ml, 
        risk_profile=risk_profile,
        user_preferences=user_preferences if user_preferences else None
    )
    return result

@router.get("/portfolio-simple", summary="Streamlined comprehensive portfolio recommendations")
def get_comprehensive_portfolio():
    """
    Streamlined endpoint that provides comprehensive portfolio recommendations
    across all asset classes using dynamic market-adapted weights.
    
    This endpoint combines stocks, mutual funds, and gold recommendations 
    with optimal allocation percentages based on current market conditions.
    """
    from app.recommender.recommender import predict_and_optimize_all_sectors
    
    result = predict_and_optimize_all_sectors()
    return result

@router.get("/budget-allocation", summary="Budget-optimized asset recommendations")
def get_budget_allocation(budget: float = Query(100000, description="Total investment budget")):
    """
    Provides budget-optimized asset recommendations across all asset classes.
    
    This endpoint allocates the specified budget proportionally based on 
    predicted ROI across stocks, mutual funds, and gold.
    
    Parameters:
    - budget: Total investment budget (default: 100,000)
    """
    from app.recommender.recommender import recommend_assets_with_budget
    
    result = recommend_assets_with_budget(total_budget=budget)
    return result

@router.get("/combined", summary="Combined asset recommendations with budget allocation")
def get_combined_recommendation(budget: float = Query(100000, description="Total investment budget")):
    """
    Combined endpoint that provides comprehensive asset recommendations with budget allocation.
    
    This endpoint combines all asset classes (stocks, mutual funds, gold) and allocates 
    the specified budget across top-performing assets based on their predicted ROI.
    
    Parameters:
    - budget: Total investment budget (default: 100,000)
    """
    from app.recommender.recommender import recommend_assets_with_budget
    
    result = recommend_assets_with_budget(total_budget=budget)
    return result

@router.get("/formatted", summary="Formatted recommendations output (human-readable)")
def get_formatted_recommendations(
    db: Session = Depends(get_db),
    top_n: int = Query(5, description="Number of recommendations per asset type"),
    use_ml: bool = Query(True, description="Use ML predictions")
):
    """
    Returns investment recommendations in a formatted, human-readable format 
    similar to backtest_runner.py output instead of raw JSON.
    
    Parameters:
    - top_n: Number of recommendations per asset type
    - use_ml: Enable ML-powered predictions
    """
    from app.recommender.recommender import recommend_assets
    import datetime
    
    # Get the raw recommendations
    result = recommend_assets(db, top_n=top_n, use_ml=use_ml, use_realtime=False)
    
    if result["status"] != "success":
        return {"error": result.get("message", "Unknown error")}
    
    # Format the output for better readability
    formatted_output = []
    formatted_output.append("=" * 80)
    formatted_output.append("🎯 SMART INVESTMENT RECOMMENDATIONS")
    formatted_output.append("=" * 80)
    formatted_output.append(f"📅 Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    formatted_output.append(f"🤖 ML Enhanced: {'Yes' if use_ml else 'No'}")
    formatted_output.append(f"📊 Total Assets Analyzed: {result.get('total_recommendations', 0)}")
    formatted_output.append("")
    
    # Format stocks
    stocks = result["recommendations"].get("stocks", [])
    if stocks:
        formatted_output.append("📈 TOP STOCK RECOMMENDATIONS:")
        formatted_output.append("-" * 50)
        for i, stock in enumerate(stocks[:top_n], 1):
            formatted_output.append(f"{i}. {stock.get('company_name', stock.get('symbol', 'Unknown'))}")
            formatted_output.append(f"   Symbol: {stock.get('symbol', 'N/A')}")
            formatted_output.append(f"   Sector: {stock.get('sector', 'N/A')}")
            formatted_output.append(f"   Predicted ROI: {stock.get('predicted_roi', 0):.2f}%")
            formatted_output.append(f"   Current Price: ₹{stock.get('current_price', 0):,.2f}")
            formatted_output.append(f"   Signal: {stock.get('signal', 'HOLD')}")
            formatted_output.append("")
    
    # Format mutual funds
    mfs = result["recommendations"].get("mutual_funds", [])
    if mfs:
        formatted_output.append("🏦 TOP MUTUAL FUND SECTORS:")
        formatted_output.append("-" * 50)
        for i, mf in enumerate(mfs[:top_n], 1):
            formatted_output.append(f"{i}. {mf.get('sector', 'Unknown')} Sector")
            formatted_output.append(f"   Score: {mf.get('score', 0):.2f}")
            formatted_output.append(f"   Return %: {mf.get('return_percentage', 0):.2f}%")
            formatted_output.append(f"   Volatility: {mf.get('volatility', 0):.2f}")
            formatted_output.append(f"   Signal: {mf.get('signal', 'HOLD')}")
            formatted_output.append("")
    
    # Format gold
    gold = result["recommendations"].get("gold", {})
    if gold:
        formatted_output.append("🥇 GOLD INVESTMENT:")
        formatted_output.append("-" * 50)
        formatted_output.append(f"   Asset Type: {gold.get('asset_type', 'Gold')}")
        formatted_output.append(f"   Score: {gold.get('score', 0):.2f}")
        formatted_output.append(f"   Signal: {gold.get('signal', 'HOLD')}")
        formatted_output.append("")
    
    # Add summary
    formatted_output.append("=" * 80)
    formatted_output.append("📋 INVESTMENT SUMMARY:")
    formatted_output.append("=" * 80)
    if stocks:
        best_stock = stocks[0]
        formatted_output.append(f"🥇 Best Stock: {best_stock.get('company_name', 'Unknown')} (ROI: {best_stock.get('predicted_roi', 0):.2f}%)")
    if mfs:
        best_mf = mfs[0]
        formatted_output.append(f"🏆 Best Sector: {best_mf.get('sector', 'Unknown')} (Score: {best_mf.get('score', 0):.2f})")
    
    formatted_output.append("")
    formatted_output.append("💡 Note: This is AI-generated advice. Please consult a financial advisor.")
    formatted_output.append("=" * 80)
    
    return {
        "formatted_output": "\n".join(formatted_output),
        "raw_data": result,
        "timestamp": datetime.datetime.now().isoformat()
    }