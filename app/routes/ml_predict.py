from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import get_db
from app.ml.predictor import (
    train_and_save_ensemble_models, 
    predict_returns,
    recommend_stocks,
    recommend_mutual_funds,
    recommend_gold,
    recommend_stocks_enhanced,
    recommend_mutual_funds_enhanced,
    recommend_gold_enhanced,
    recommend_all_assets_enhanced
)

router = APIRouter()

@router.post("/train-model/{asset_type}")
def train_model_endpoint(asset_type: str, db: Session = Depends(get_db)):
    """Train machine learning model for investment prediction"""
    valid_types = ["stocks", "mutual_funds", "gold"]
    if asset_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Asset type must be one of: {valid_types}")
    
    try:
        result = train_and_save_ensemble_models(db, asset_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.get("/predict/{asset_type}")
def predict_endpoint(asset_type: str, db: Session = Depends(get_db)):
    """Get predictions using trained model (placeholder - requires model storage)"""
    # Note: This is a simplified version. In production, you'd store/load trained models
    try:
        # For demo purposes, let's just return model training results
        result = train_and_save_ensemble_models(db, asset_type)
        return {
            "status": "demo_prediction",
            "message": "This endpoint shows training results. In production, you'd use stored models.",
            "training_results": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/recommend/stocks")
def recommend_stocks_endpoint(top_n: int = 5, db: Session = Depends(get_db)):
    """Get ML-based stock sector recommendations"""
    try:
        recommendations = recommend_stocks(db, top_n)
        return {
            "status": "success",
            "asset_type": "stocks",
            "recommendations": recommendations,
            "count": len(recommendations),
            "method": "ML-based predictions with weighted scoring"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stock recommendation failed: {str(e)}")

@router.get("/recommend/mutual-funds")
def recommend_mutual_funds_endpoint(top_n: int = 5, db: Session = Depends(get_db)):
    """Get ML-based mutual fund sector recommendations"""
    try:
        recommendations = recommend_mutual_funds(db, top_n)
        return {
            "status": "success",
            "asset_type": "mutual_funds",
            "recommendations": recommendations,
            "count": len(recommendations),
            "method": "ML-based predictions with weighted scoring"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mutual fund recommendation failed: {str(e)}")

@router.get("/recommend/gold")
def recommend_gold_endpoint(top_n: int = 3, db: Session = Depends(get_db)):
    """Get ML-based gold investment recommendations"""
    try:
        recommendations = recommend_gold(db, top_n)
        return {
            "status": "success",
            "asset_type": "gold",
            "recommendations": recommendations,
            "count": len(recommendations),
            "method": "ML-based predictions with weighted scoring"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gold recommendation failed: {str(e)}")

@router.get("/recommend/stocks/enhanced")
def recommend_stocks_enhanced_endpoint(top_n: int = 5, db: Session = Depends(get_db)):
    """Get enhanced ML-based stock recommendations with conversational explanations"""
    try:
        return recommend_stocks_enhanced(db, top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced stock recommendation failed: {str(e)}")

@router.get("/recommend/mutual-funds/enhanced")
def recommend_mutual_funds_enhanced_endpoint(top_n: int = 5, db: Session = Depends(get_db)):
    """Get enhanced ML-based mutual fund recommendations with conversational explanations"""
    try:
        return recommend_mutual_funds_enhanced(db, top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced mutual fund recommendation failed: {str(e)}")

@router.get("/recommend/gold/enhanced")
def recommend_gold_enhanced_endpoint(top_n: int = 3, db: Session = Depends(get_db)):
    """Get enhanced ML-based gold recommendations with conversational explanations"""
    try:
        return recommend_gold_enhanced(db, top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced gold recommendation failed: {str(e)}")

@router.get("/recommend/all/enhanced")
def recommend_all_enhanced_endpoint(top_n: int = 3, db: Session = Depends(get_db)):
    """Get enhanced recommendations for all asset types with conversational explanations"""
    try:
        stocks = recommend_stocks_enhanced(db, top_n)
        mutual_funds = recommend_mutual_funds_enhanced(db, top_n)
        gold = recommend_gold_enhanced(db, top_n)
        
        return {
            "status": "success",
            "date": "2025-08-02",
            "strategy": "ML-based Multi-Asset Recommendations with Conversational Explanations",
            "assets_covered": ["stocks", "mutual_funds", "gold"],
            "stocks": stocks,
            "mutual_funds": mutual_funds,
            "gold": gold,
            "overall_summary": {
                "total_recommendations": len(stocks['recommendations']) + len(mutual_funds['recommendations']) + len(gold['recommendations']),
                "asset_class_diversification": 3,
                "method": "Enhanced ML predictions with timing analysis and conversational explanations"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced comprehensive recommendation failed: {str(e)}")
