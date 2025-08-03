import pandas as pd
from app.db import get_db_connection
from app.ml.predictor import full_recommendation

def load_data_from_db(table_name: str) -> pd.DataFrame:
    conn = get_db_connection()
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df.dropna().reset_index(drop=True)

def run_all_recommendations(total_budget: float = 100000):
    # Divide budget across sectors
    stock_budget = total_budget * 0.4
    mf_budget = total_budget * 0.3
    gold_budget = total_budget * 0.3

    print("🔄 Loading data and generating predictions...")

    stock_data = load_data_from_db("stock_sector_analysis")
    mf_data = load_data_from_db("mutual_fund_sector_analysis")
    gold_data = load_data_from_db("gold_sector_analysis")

    print("\n📈 Running stock recommendation...")
    stock_result = full_recommendation(stock_data, stock_budget)

    print("\n📊 Running mutual fund recommendation...")
    mf_result = full_recommendation(mf_data, mf_budget)

    print("\n🏅 Running gold recommendation...")
    gold_result = full_recommendation(gold_data, gold_budget)

    return {
        "stocks": stock_result,
        "mutual_funds": mf_result,
        "gold": gold_result
    }

if __name__ == "__main__":
    results = run_all_recommendations(100000)

    print("\n✅ Final Recommended Portfolio:")
    print("\n📈 Stocks:")
    print(results["stocks"])
    
    print("\n📊 Mutual Funds:")
    print(results["mutual_funds"])
    
    print("\n🏅 Gold:")
    print(results["gold"])