# app/utils/historical_returns.py

import pandas as pd
import psycopg2
from datetime import datetime
from app.config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD


def fetch_sector_returns(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical daily percentage returns for Stocks, Mutual Funds, and Gold from PostgreSQL.
    Returns a DataFrame with columns: [date, stocks, mutual_funds, gold]
    """
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    
    query_template = """
        SELECT date, adjusted_close 
        FROM {table} 
        WHERE date BETWEEN %s AND %s
        ORDER BY date
    """

    tables = {
        "stocks": "stock_daily_prices",
        "mutual_funds": "mutual_fund_daily_prices",
        "gold": "gold_daily_prices"
    }

    data = {}

    for sector, table in tables.items():
        df = pd.read_sql(query_template.format(table=table), conn, params=(start_date, end_date))
        df = df.sort_values("date")
        df[sector] = df["adjusted_close"].pct_change()
        df = df.drop(columns=["adjusted_close"])
        data[sector] = df

    conn.close()

    # Merge on date
    merged = data["stocks"]
    merged = merged.merge(data["mutual_funds"], on="date", how="outer")
    merged = merged.merge(data["gold"], on="date", how="outer")
    
    merged = merged.dropna().reset_index(drop=True)
    return merged