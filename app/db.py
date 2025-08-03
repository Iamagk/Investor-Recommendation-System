import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://postgres:pass@localhost:5432/postgres"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_postgres_connection():
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="pass",
            host="localhost",
            port="5432"
        )
        return conn
    except Exception as e:
        print("Error connecting to PostgreSQL:", e)
        return None

def insert_stock_data(symbol, name, current_price, market_cap, pe_ratio, sector):
    conn = get_postgres_connection()
    if conn:
        cur = conn.cursor()
        query = """
            INSERT INTO stocks (symbol, name, current_price, market_cap, pe_ratio, sector)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cur.execute(query, (symbol, name, current_price, market_cap, pe_ratio, sector))
        conn.commit()
        cur.close()
        conn.close()

# You can later add:
# - insert_mutual_fund_data()
# - insert_gold_data()