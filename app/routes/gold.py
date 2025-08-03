from fastapi import APIRouter
from app.services.gold_service import store_gold_rates

router = APIRouter()

@router.get("/")
def get_gold():
    """Get gold recommendations"""
    return {"message": "Gold recommendations endpoint"}

@router.get("/store-rates")
def store_current_gold_rates():
    """Store current gold rates in database"""
    try:
        from data.fetch_gold import get_gold_rates
        
        gold_df = get_gold_rates()
        if gold_df is not None and not gold_df.empty:
            store_gold_rates(gold_df)
            return {
                "status": "success",
                "message": "Gold rates stored successfully",
                "count": len(gold_df)
            }
        else:
            return {
                "status": "error",
                "message": "No gold rate data to store"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error storing gold rates: {str(e)}"
        }

@router.get("/recommendations")
def get_gold_recommendations():
    """Get personalized gold investment recommendations"""
    return {"message": "Gold recommendations will be implemented here"}

@router.get("/prices")
def get_gold_prices():
    """Get current gold prices"""
    try:
        from data.fetch_gold import get_gold_rates
        
        gold_df = get_gold_rates()
        if gold_df is not None and not gold_df.empty:
            gold_data = gold_df.to_dict('records')
            return {
                "status": "success",
                "message": "Current gold prices",
                "count": len(gold_data),
                "data": gold_data
            }
        else:
            return {
                "status": "error", 
                "message": "No gold price data available"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error fetching gold prices: {str(e)}"
        }
