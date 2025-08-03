import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = returns / std
    return returns, std, sharpe_ratio


def negative_sharpe_ratio(weights, mean_returns, cov_matrix):
    _, _, sharpe = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    return -sharpe


def calculate_optimal_portfolio(returns_df: pd.DataFrame) -> dict:
    """
    returns_df: DataFrame where each column is a sector/asset and rows are historical % returns
    returns_df =
              StockA   StockB   Gold   MutualFund
    2024-01   0.01     0.02     0.015     0.01
    2024-02   0.005    -0.01    0.02      0.008
    ...
    """
    assets = returns_df.columns.tolist()
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    num_assets = len(assets)

    # Initial guess and constraints
    init_guess = num_assets * [1. / num_assets]
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    result = minimize(
        negative_sharpe_ratio,
        init_guess,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    allocation = {asset: round(weight, 4) for asset, weight in zip(assets, optimal_weights)}
    return allocation