import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.db import get_db
from app.models.analysis_models import (
    StockSectorAnalysis,
    ComprehensiveSectorAnalysis,
    EnhancedSectorScores
)

router = APIRouter()

@router.post("/load-analysis-data")
def load_analysis_data(db: Session = Depends(get_db)):
    try:
        # Load CSV files
        stock_df = pd.read_csv("data/stock_sector_analysis_20250801_143502.csv")
        comp_df = pd.read_csv("data/comprehensive_sector_analysis_20250801_143502.csv")
        enhanced_df = pd.read_csv("data/enhanced_sector_scores_20250801_143501.csv")

        # Load stock sector analysis
        db.query(StockSectorAnalysis).delete()
        for _, row in stock_df.iterrows():
            db.add(StockSectorAnalysis(**row.to_dict()))

        # Load comprehensive sector analysis
        db.query(ComprehensiveSectorAnalysis).delete()
        for _, row in comp_df.iterrows():
            db.add(ComprehensiveSectorAnalysis(**row.to_dict()))

        # Load enhanced sector scores
        db.query(EnhancedSectorScores).delete()
        for _, row in enhanced_df.iterrows():
            db.add(EnhancedSectorScores(**row.to_dict()))

        db.commit()
        return {"status": "success", "message": "Analysis data loaded successfully."}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))