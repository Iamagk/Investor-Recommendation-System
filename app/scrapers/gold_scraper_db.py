import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.db import get_postgres_connection

def insert_gold_data(data):
    """Insert gold price data into PostgreSQL database"""
    if data is None:
        print("No gold data provided")
        return False
        
    # Handle DataFrame input
    if hasattr(data, 'empty') and data.empty:
        print("Empty gold DataFrame provided")
        return False
        
    # Convert DataFrame to list of dictionaries
    if hasattr(data, 'to_dict'):
        try:
            data_list = data.to_dict('records')
        except Exception as e:
            print(f"Error converting DataFrame to records: {e}")
            return False
    elif isinstance(data, list):
        data_list = data
    else:
        print(f"Unsupported data type for gold data: {type(data)}")
        return False
    
    if not data_list or len(data_list) == 0:
        print("No gold data records to insert")
        return False
    
    conn = None
    cursor = None
    
    try:
        conn = get_postgres_connection()
        if not conn:
            print("Failed to connect to PostgreSQL")
            return False
            
        cursor = conn.cursor()
        inserted_count = 0

        for gold_record in data_list:
            try:
                # Extract data from the record with flexible field mapping
                date = (gold_record.get("Date") or 
                       gold_record.get("date") or 
                       "2025-08-02")  # Default to today
                
                # Try multiple price field names
                price = (gold_record.get("Price_per_gram") or
                        gold_record.get("price_per_gram") or 
                        gold_record.get("Price") or 
                        gold_record.get("price") or 
                        gold_record.get("Current Price") or 
                        gold_record.get("current_price") or
                        gold_record.get("22K (INR/gm)") or
                        gold_record.get("24K (INR/gm)"))
                
                # Try multiple source field names  
                source = (gold_record.get("Source") or 
                         gold_record.get("source") or 
                         gold_record.get("Name") or 
                         gold_record.get("Symbol") or 
                         "Unknown")
                
                # Skip if essential data is missing
                if not price:
                    print(f"Skipping record with no price: {gold_record}")
                    continue
                    
                # Validate price is numeric and reasonable
                try:
                    price_float = float(price)
                    if price_float <= 0 or price_float > 100000:  # Reasonable bounds
                        print(f"Skipping record with unreasonable price {price_float}: {gold_record}")
                        continue
                except (ValueError, TypeError):
                    print(f"Skipping record with invalid price {price}: {gold_record}")
                    continue
                    
                # Check if record already exists for this date and source
                cursor.execute("SELECT id FROM gold_prices WHERE date = %s AND source = %s", (date, source))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing record
                    cursor.execute("""
                        UPDATE gold_prices SET
                            price_per_gram = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE date = %s AND source = %s;
                    """, (price_float, date, source))
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO gold_prices (date, price_per_gram, source)
                        VALUES (%s, %s, %s);
                    """, (date, price_float, source))
                
                inserted_count += 1
                
            except Exception as record_error:
                print(f"Error processing gold record {gold_record}: {record_error}")
                continue

        conn.commit()
        print(f"Successfully inserted/updated {inserted_count} gold price records")
        return inserted_count > 0
            
        conn.commit()
        print(f"Successfully inserted/updated {inserted_count} gold price records")
        return inserted_count > 0
        
    except Exception as e:
        print(f"Error inserting gold data: {e}")
        if conn:
            conn.rollback()
        return False
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_gold_data_from_db():
    """Retrieve gold data from PostgreSQL database"""
    try:
        conn = get_postgres_connection()
        if not conn:
            return []
            
        cursor = conn.cursor()
        cursor.execute("""
            SELECT date, price_per_gram, source, updated_at
            FROM gold_prices 
            ORDER BY updated_at DESC
            LIMIT 100
        """)
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        gold_data = []
        for row in rows:
            gold_data.append({
                "date": row[0].isoformat() if row[0] else None,
                "price_per_gram": float(row[1]) if row[1] else 0.0,
                "source": row[2],
                "updated_at": row[3].isoformat() if row[3] else None
            })
        
        return gold_data
        
    except Exception as e:
        print(f"Error retrieving gold data: {e}")
        return []
