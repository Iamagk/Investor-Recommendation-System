import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.db import get_postgres_connection

def insert_stock_data(data):
    """Insert stock data into the database with proper error handling"""
    if not data or len(data) == 0:
        print("No stock data to insert")
        return False
    
    conn = None
    cursor = None
    
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()

        for stock in data:
            # Check if stock already exists
            cursor.execute("SELECT id FROM stocks WHERE symbol = %s", (stock.get("symbol", ""),))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                cursor.execute("""
                    UPDATE stocks SET
                        name = %s,
                        sector = %s,
                        last_price = %s,
                        change_percent = %s,
                        market_cap = %s,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE symbol = %s;
                """, (
                    stock.get("name", stock.get("companyName", "")),
                    stock.get("sector", stock.get("industry", "N/A")),
                    stock.get("last_price", stock.get("lastPrice", 0)),
                    stock.get("change_percent", stock.get("pChange", 0)),
                    stock.get("market_cap", stock.get("marketCap", None)),
                    stock.get("symbol", "")
                ))
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO stocks (symbol, name, sector, last_price, change_percent, market_cap)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """, (
                    stock.get("symbol", ""),
                    stock.get("name", stock.get("companyName", "")),
                    stock.get("sector", stock.get("industry", "N/A")),
                    stock.get("last_price", stock.get("lastPrice", 0)),
                    stock.get("change_percent", stock.get("pChange", 0)),
                    stock.get("market_cap", stock.get("marketCap", None))
                ))

        conn.commit()
        print(f"Successfully inserted/updated {len(data)} stock records")
        return True
        
    except Exception as e:
        print(f"Error inserting stock data: {e}")
        if conn:
            conn.rollback()
        return False
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_stock_data_from_db(limit=100):
    """Retrieve stock data from the database"""
    conn = None
    cursor = None
    
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT symbol, name, sector, last_price, day_high, day_low, 
                   volume, market_cap, pe_ratio, updated_at
            FROM stocks
            ORDER BY updated_at DESC
            LIMIT %s;
        """, (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        
        # Convert to list of dictionaries
        stock_data = []
        for row in results:
            stock_dict = dict(zip(columns, row))
            stock_data.append(stock_dict)
            
        return stock_data
        
    except Exception as e:
        print(f"Error retrieving stock data: {e}")
        return []
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()