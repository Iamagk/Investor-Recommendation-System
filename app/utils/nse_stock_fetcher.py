import pandas as pd
import requests
from io import StringIO

NSE_EQUITY_CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

def fetch_nse_equities():
    """Fetch NSE equity list with timeout and fallback"""
    try:
        # Add timeout and better error handling
        response = requests.get(NSE_EQUITY_CSV_URL, timeout=10)
        response.raise_for_status()
        
        # Use StringIO to read CSV from response text
        df = pd.read_csv(StringIO(response.text))
        df = df[['SYMBOL', 'NAME OF COMPANY', 'ISIN NUMBER']]
        df.columns = ['symbol', 'company', 'isin']
        df = df.dropna()
        
        return df.to_dict(orient="records")
    except requests.exceptions.Timeout:
        return {"error": "Request timeout - NSE server is slow"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error - Cannot reach NSE server"}
    except pd.errors.EmptyDataError:
        return {"error": "Empty or invalid CSV data received"}
    except Exception as e:
        return {"error": f"Failed to fetch NSE equities: {str(e)}"}