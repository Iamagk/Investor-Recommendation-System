import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calculate_portfolio_performance(weights, returns, cov_matrix):
    expected_return = np.dot(weights, returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = expected_return / volatility if volatility != 0 else 0
    return expected_return, volatility, sharpe_ratio


def negative_sharpe_ratio(weights, returns, cov_matrix):
    _, _, sharpe = calculate_portfolio_performance(weights, returns, cov_matrix)
    return -sharpe  # Minimizing the negative Sharpe ratio


def optimize_portfolio_allocations(asset_data: pd.DataFrame):
    """
    asset_data: pd.DataFrame with columns ['Asset', 'ExpectedReturn', 'Volatility']
    Returns: dict of asset allocations (e.g., {'Stocks': 0.4, 'Mutual Funds': 0.3, 'Gold': 0.3})
    """
    assets = asset_data['Asset'].tolist()
    returns = asset_data['ExpectedReturn'].values
    volatilities = asset_data['Volatility'].values

    # Assume diagonal covariance (independent assets)
    cov_matrix = np.diag(volatilities ** 2)

    num_assets = len(assets)
    initial_guess = np.array([1.0 / num_assets] * num_assets)
    bounds = [(0.0, 1.0)] * num_assets
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights must sum to 1

    result = minimize(
        fun=negative_sharpe_ratio,
        x0=initial_guess,
        args=(returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimized_weights = result.x
    allocation = {assets[i]: float(round(optimized_weights[i], 4)) for i in range(num_assets)}
    return allocation


# Optional: helper for testing
def example_run():
    data = pd.DataFrame({
        'Asset': ['Stocks', 'Mutual Funds', 'Gold'],
        'ExpectedReturn': [0.12, 0.08, 0.06],  # Annualized
        'Volatility': [0.2, 0.1, 0.15]  # Std deviation
    })
    allocation = optimize_portfolio_allocations(data)
    print("Optimized Allocation:", allocation)


if __name__ == "__main__":
    example_run()