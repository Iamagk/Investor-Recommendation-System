import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.db import get_postgres_connection

def insert_mutual_fund_data(data):
    """Insert mutual fund data into the database with proper error handling"""
    if not data or len(data) == 0:
        print("No mutual fund data to insert")
        return False
    
    conn = None
    cursor = None
    
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()

        for mf in data:
            fund_name = mf.get("fund_name", mf.get("scheme_name", ""))
            
            # Check if mutual fund already exists
            cursor.execute("SELECT id FROM mutual_funds WHERE fund_name = %s", (fund_name,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                cursor.execute("""
                    UPDATE mutual_funds SET
                        fund_house = %s,
                        category = %s,
                        nav = %s,
                        returns_1y = %s,
                        returns_3y = %s,
                        risk_level = %s,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE fund_name = %s;
                """, (
                    mf.get("fund_house", "Unknown"),
                    mf.get("category", "N/A"),
                    mf.get("nav", 0),
                    mf.get("returns_1y", mf.get("returns_1yr", None)),
                    mf.get("returns_3y", mf.get("returns_3yr", None)),
                    mf.get("risk_level", "Medium"),
                    fund_name
                ))
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO mutual_funds (fund_name, fund_house, category, nav, returns_1y, returns_3y, risk_level)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                """, (
                    fund_name,
                    mf.get("fund_house", "Unknown"),
                    mf.get("category", "N/A"),
                    mf.get("nav", 0),
                    mf.get("returns_1y", mf.get("returns_1yr", None)),
                    mf.get("returns_3y", mf.get("returns_3yr", None)),
                    mf.get("risk_level", "Medium")
                ))

        conn.commit()
        print(f"Successfully inserted/updated {len(data)} mutual fund records")
        return True
        
    except Exception as e:
        print(f"Error inserting mutual fund data: {e}")
        if conn:
            conn.rollback()
        return False
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_mutual_fund_data_from_db(limit=100, category=None):
    """Retrieve mutual fund data from the database"""
    conn = None
    cursor = None
    
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()
        
        if category:
            cursor.execute("""
                SELECT scheme_code, scheme_name, category, nav, nav_date, 
                       returns_1yr, returns_3yr, returns_5yr, updated_at
                FROM mutual_funds
                WHERE category ILIKE %s
                ORDER BY nav DESC
                LIMIT %s;
            """, (f"%{category}%", limit))
        else:
            cursor.execute("""
                SELECT scheme_code, scheme_name, category, nav, nav_date, 
                       returns_1yr, returns_3yr, returns_5yr, updated_at
                FROM mutual_funds
                ORDER BY updated_at DESC
                LIMIT %s;
            """, (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        
        # Convert to list of dictionaries
        mf_data = []
        for row in results:
            mf_dict = dict(zip(columns, row))
            mf_data.append(mf_dict)
            
        return mf_data
        
    except Exception as e:
        print(f"Error retrieving mutual fund data: {e}")
        return []
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_top_performing_funds(limit=10, period="1yr"):
    """Get top performing mutual funds by returns"""
    conn = None
    cursor = None
    
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()
        
        # Choose the returns column based on period
        returns_column = f"returns_{period}"
        if returns_column not in ["returns_1yr", "returns_3yr", "returns_5yr"]:
            returns_column = "returns_1yr"
        
        cursor.execute(f"""
            SELECT scheme_code, scheme_name, category, nav, nav_date, 
                   returns_1yr, returns_3yr, returns_5yr, updated_at
            FROM mutual_funds
            WHERE {returns_column} IS NOT NULL
            ORDER BY {returns_column} DESC
            LIMIT %s;
        """, (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        
        # Convert to list of dictionaries
        top_funds = []
        for row in results:
            fund_dict = dict(zip(columns, row))
            top_funds.append(fund_dict)
            
        return top_funds
        
    except Exception as e:
        print(f"Error retrieving top performing funds: {e}")
        return []
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()