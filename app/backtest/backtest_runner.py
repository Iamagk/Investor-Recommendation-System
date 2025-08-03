import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Example file (this should be dynamic later)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.utils.file_utils import get_latest_analysis_csv

# Use the latest CSV file instead of hardcoded date
CSV_FILE = get_latest_analysis_csv('stock') or "data/stock_sector_analysis_20250801_143502.csv"


def load_stock_data():
    """Load sector analysis data and simulate time series for backtesting"""
    df = pd.read_csv(CSV_FILE)
    
    # Since we don't have time series data, let's simulate it
    # Create a date range for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Expand the sector data across the date range
    expanded_data = []
    for date in date_range:
        for _, row in df.iterrows():
            # Simulate daily returns based on momentum and volatility
            base_return = row['avg_return_pct']
            volatility = row['volatility'] if row['volatility'] > 0 else 0.1
            momentum = row['momentum_score']
            
            # Add some randomness based on volatility
            daily_return = base_return * (1 + np.random.normal(0, volatility * 0.1))
            
            expanded_data.append({
                'date': date,
                'sector': row['sector'],
                'return_pct': daily_return,
                'momentum_score': momentum,
                'volatility': volatility,
                'investment_count': row['investment_count'],
                'avg_price': row['avg_price']
            })
    
    expanded_df = pd.DataFrame(expanded_data)
    expanded_df['date'] = pd.to_datetime(expanded_df['date'])
    return expanded_df.sort_values(by='date')


def backtest_strategy(df, top_n=3, hold_days=7):
    """Backtest strategy using sector data"""
    start_date = df['date'].min()
    end_date = df['date'].max()
    
    portfolio_returns = []
    current_date = start_date

    while current_date + timedelta(days=hold_days) <= end_date:
        # Filter current day's data
        current_day_data = df[df['date'] == current_date]
        
        # Pick top N sectors with highest momentum score (our prediction metric)
        top_sectors = current_day_data.sort_values(by='momentum_score', ascending=False).head(top_n)

        # Calculate actual return after holding period
        hold_end_date = current_date + timedelta(days=hold_days)
        future_data = df[df['date'] == hold_end_date]

        returns = []
        selected_sectors = []
        for _, row in top_sectors.iterrows():
            sector = row['sector']
            future_row = future_data[future_data['sector'] == sector]
            if not future_row.empty:
                actual_roi = future_row.iloc[0]['return_pct']
                returns.append(actual_roi)
                selected_sectors.append(sector)

        if returns:
            avg_return = np.mean(returns)
            portfolio_returns.append({
                'date': current_date,
                'avg_return': avg_return,
                'sectors': selected_sectors,
                'individual_returns': returns
            })

        current_date += timedelta(days=hold_days)

    return pd.DataFrame(portfolio_returns)


def evaluate_performance(portfolio_df):
    returns = portfolio_df['avg_return']
    
    cumulative_return = (1 + returns).prod() - 1
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(52) if returns.std() > 0 else 0
    hit_ratio = sum(returns > 0) / len(returns)
    max_drawdown = ((returns.cumsum().cummax() - returns.cumsum()).max())

    return {
        'Cumulative Return': cumulative_return,
        'Sharpe Ratio': sharpe_ratio,
        'Hit Ratio': hit_ratio,
        'Max Drawdown': max_drawdown,
        'Total Trades': len(returns)
    }


if __name__ == "__main__":
    df = load_stock_data()
    result_df = backtest_strategy(df)
    metrics = evaluate_performance(result_df)

    print("\nBacktest Summary:\n")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nTrade log preview:")
    print(result_df.head())