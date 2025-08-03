from fastapi import APIRouter
import pandas as pd
from app.db import get_postgres_connection

router = APIRouter()

def load_csv_to_table(csv_path, table_name, conn):
    try:
        df = pd.read_csv(csv_path)
        print(f"Loading {len(df)} rows into {table_name}")
        print(f"CSV columns: {list(df.columns)}")
        
        # Special handling for enhanced_sector_scores with column mapping
        if table_name == "enhanced_sector_scores":
            # Map CSV columns to database columns
            column_mapping = {
                'avg_return_%': 'avg_return_percent',
                'avg_volatility_%': 'avg_volatility_percent'
            }
            df = df.rename(columns=column_mapping)
        
        # Clean column names - remove special characters and replace with underscores
        df.columns = [c.lower().replace('%', '_percent').replace(' ', '_').replace('-', '_') for c in df.columns]
        print(f"Final columns: {list(df.columns)}")
        
        cur = conn.cursor()
        
        # Clear existing data first
        cur.execute(f"DELETE FROM {table_name}")
        print(f"Cleared existing data from {table_name}")
        
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                placeholders = ','.join(['%s'] * len(row))
                columns = ','.join(row.index)
                sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                cur.execute(sql, tuple(row))
            except Exception as e:
                print(f"Error inserting row {i}: {row.to_dict()}")
                print(f"SQL: {sql}")
                print(f"Error: {str(e)}")
                raise e
        
        conn.commit()
        cur.close()
        print(f"Successfully loaded {len(df)} rows into {table_name}")
        
    except Exception as e:
        print(f"Error loading {csv_path} into {table_name}: {str(e)}")
        if 'cur' in locals():
            cur.close()
        raise e

@router.post("/load-analysis-data/")
def load_all_analysis_data():
    try:
        conn = get_postgres_connection()
        from app.utils.file_utils import get_all_latest_analysis_csvs
        
        # Get latest CSV files
        latest_csvs = get_all_latest_analysis_csvs()
        
        # Load the latest CSV files with fallback to hardcoded ones
        comprehensive_csv = latest_csvs['comprehensive'] or "data/comprehensive_sector_analysis_20250801_143502.csv"
        enhanced_csv = latest_csvs['enhanced'] or "data/enhanced_sector_scores_20250801_143501.csv"  
        stock_csv = latest_csvs['stock'] or "data/stock_sector_analysis_20250801_143502.csv"
        
        load_csv_to_table(comprehensive_csv, "comprehensive_sector_analysis", conn)
        load_csv_to_table(enhanced_csv, "enhanced_sector_scores", conn)
        load_csv_to_table(stock_csv, "stock_sector_analysis", conn)
        conn.close()
        return {"status": "success", "message": "All data loaded into PostgreSQL."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/analysis-data")
def get_analysis_data():
    """Get count of loaded analysis data"""
    try:
        conn = get_postgres_connection()
        cur = conn.cursor()
        
        # Get counts from each table
        results = {}
        tables = ["stock_sector_analysis", "comprehensive_sector_analysis", "enhanced_sector_scores"]
        
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            results[table] = count
        
        cur.close()
        conn.close()
        
        return {
            "status": "success",
            "data": results
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}