# services/gold_service.py

from pymongo import MongoClient
from datetime import datetime

def store_gold_rates(gold_df, mongo_uri="mongodb://localhost:27017", db_name="investment_advisor"):
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db["gold_rates"]

        gold_df["timestamp"] = datetime.utcnow()

        # Convert to dictionary and insert
        records = gold_df.to_dict(orient="records")
        collection.insert_many(records)

        print(f"Inserted {len(records)} gold rate records into MongoDB.")

    except Exception as e:
        print("Error while storing gold data:", e)