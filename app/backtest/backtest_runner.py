import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Example file (this should be dynamic later)
CSV_FILE = "data/stock_sector_analysis_20250801_143502.csv"


def load_stock_data():
    df = pd.read_csv(CSV_FILE)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(by='date')


def backtest_strategy(df, top_n=3, hold_days=7):
    start_date = df['date'].min()
    end_date = df['date'].max()
    
    portfolio_returns = []
    current_date = start_date

    while current_date + timedelta(days=hold_days) <= end_date:
        # Filter current day's predictions
        current_day_data = df[df['date'] == current_date]
        
        # Pick top N assets with highest predicted ROI
        top_assets = current_day_data.sort_values(by='predicted_roi', ascending=False).head(top_n)

        # Calculate actual return after holding period
        hold_end_date = current_date + timedelta(days=hold_days)
        future_data = df[df['date'] == hold_end_date]

        returns = []
        for _, row in top_assets.iterrows():
            asset = row['symbol']
            future_row = future_data[future_data['symbol'] == asset]
            if not future_row.empty:
                actual_roi = future_row.iloc[0]['actual_roi']
                returns.append(actual_roi)

        if returns:
            avg_return = np.mean(returns)
            portfolio_returns.append({
                'date': current_date,
                'avg_return': avg_return,
                'assets': list(top_assets['symbol'])
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