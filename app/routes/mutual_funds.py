from fastapi import APIRouter
from app.services.mutual_funds_service import get_mutual_fund_data, get_all_mutual_funds

router = APIRouter(prefix="/mutual-funds", tags=["Mutual Funds"])

@router.get("/navs", summary="Get latest NAVs of selected mutual funds")
def navs():
    return get_mutual_fund_data()

@router.get("/all")
def fetch_all_mutual_funds():
    return get_all_mutual_funds()

@router.get("/status")
def get_mutual_funds_status():
    """Quick status check for mutual funds endpoints"""
    return {
        "service": "mutual_funds", 
        "status": "active",
        "available_endpoints": {
            "/mutual-funds": "Real AMFI mutual fund data",
            "/mutual-funds/navs": "AMFI NAV data from database",
            "/mutual-funds/all": "All mutual funds from database"
        }
    }
