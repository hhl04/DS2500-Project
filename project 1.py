import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, percentileofscore
import matplotlib.pyplot as plt
import time
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    macd_hist = macd - signal

    return macd, signal, macd_hist

# Function to calculate SMA
def calculate_sma(data, window):
    return data.rolling(window=window).mean()

# Function to get stock data and calculate metrics
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    # Get historical data for calculating technical indicators
    hist = stock.history(period="1y")

    return {
        'Ticker': ticker,
        'P/E Ratio': info.get('trailingPE', np.nan),
        'Forward P/E': info.get('forwardPE', np.nan),
        'PEG Ratio': info.get('pegRatio', np.nan),
        'Price to Book': info.get('priceToBook', np.nan),
        'Dividend Yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else np.nan,
        'Debt to Equity': info.get('debtToEquity', np.nan),
        'ROE': info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan,
        'ROA': info.get('returnOnAssets', np.nan) * 100 if info.get('returnOnAssets') else np.nan,
        'Revenue Growth': info.get('revenueGrowth', np.nan) * 100 if info.get('revenueGrowth') else np.nan,
        'Gross Margins': info.get('grossMargins', np.nan) * 100 if info.get('grossMargins') else np.nan,
        'Free Cash Flow': info.get('freeCashflow', np.nan),
        'Current Ratio': info.get('currentRatio', np.nan),
        'Quick Ratio': info.get('quickRatio', np.nan),
        'Profit Margin': info.get('profitMargins', np.nan) * 100 if info.get('profitMargins') else np.nan,
        'RSI': calculate_rsi(hist['Close'])[-1] if len(hist) > 14 else np.nan,
        'MACD': calculate_macd(hist['Close'])[0][-1] if len(hist) > 26 else np.nan,
        '50D SMA': calculate_sma(hist['Close'], 50)[-1] if len(hist) > 50 else np.nan,
        '200D SMA': calculate_sma(hist['Close'], 200)[-1] if len(hist) > 200 else np.nan,
    }

# Define scoring metrics and thresholds
metrics = {
    'P/E Ratio': (0, 50),
    'Forward P/E': (0, 50),
    'PEG Ratio': (0, 50),
    'Price to Book': (0, 50),
    'Dividend Yield': (50, 100),
    'Debt to Equity': (0, 50),
    'ROE': (50, 100),
    'ROA': (50, 100),
    'Revenue Growth': (50, 100),
    'Gross Margins': (50, 100),
    'Free Cash Flow': (50, 100),
    'Current Ratio': (50, 100),
    'Quick Ratio': (50, 100),
    'Profit Margin': (50, 100),
    'RSI': (30, 70),  # Avoid overbought or oversold
    'MACD': (50, 100),  # Prefer upward trend
}

# Function to score stocks based on metrics
def score_stock(row, metrics):
    score = 0
    for metric, (min_percentile, max_percentile) in metrics.items():
        if pd.notna(row[metric]):
            percentile = percentileofscore(df[metric].dropna(), row[metric])
            if min_percentile <= percentile <= max_percentile:
                score += 1
    return score

# Function to calculate Sharpe Ratio for optimization
def sharpe_ratio(weights, returns, cov_matrix):
    portfolio_return = np.sum(returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -portfolio_return / portfolio_volatility

# Function to optimize portfolio weights based on Sharpe Ratio
def optimize_portfolio(returns):
    n = len(returns.columns)
    cov_matrix = returns.cov() * 252  # Annualize the covariance matrix
    args = (returns.mean() * 252, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))

    result = minimize(sharpe_ratio, np.array([1 / n] * n), args=args, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    return result.x

# Function to build and optimize portfolio based on risk tolerance
def build_and_optimize_portfolio(tickers, risk_tolerance, metrics):
    global df
    # Step 1: Fetch data and score stocks
    stock_data = [get_stock_data(ticker) for ticker in tickers]
    df = pd.DataFrame(stock_data)
    df['Score'] = df.apply(lambda row: score_stock(row, metrics), axis=1)

    # Step 2: Select top stocks based on risk tolerance and scores
    if risk_tolerance == 'low':
        selected_stocks = df.nlargest(5, 'Score')['Ticker'].tolist()
    elif risk_tolerance == 'medium':
        selected_stocks = df.nlargest(10, 'Score')['Ticker'].tolist()
    else:  # high risk tolerance
        selected_stocks = df.nlargest(15, 'Score')['Ticker'].tolist()

    # Step 3: Optimize portfolio weights
    data = yf.download(selected_stocks, start='2023-01-01', end='2024-01-01')
    daily_returns = data['Adj Close'].pct_change().dropna()
    annual_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252

    optimized_weights = optimize_portfolio(daily_returns)

    return selected_stocks, optimized_weights

# Function to rebalance the portfolio
def rebalance_portfolio(tickers, risk_tolerance, metrics):
    selected_stocks, optimal_weights = build_and_optimize_portfolio(tickers, risk_tolerance, metrics)
    print("\nRebalanced Portfolio:")
    for stock, weight in zip(selected_stocks, optimal_weights):
        print(f"{stock}: {weight:.4f}")
    return selected_stocks, optimal_weights

# Function to backtest the portfolio
def backtest(stocks, start_date, end_date, rebalance_frequency='M'):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    data = yf.download(stocks, start=start_date, end=end_date)['Adj Close']

    if data.empty:
        print("No data downloaded for the selected stocks.")
        return

    returns = data.pct_change().dropna()
    portfolio_value = 1
    portfolio_returns = []
    weights = np.array([1 / len(stocks)] * len(stocks))  # Initial equal weights

    for date, row in returns.resample(rebalance_frequency).first().iterrows():
        period_return = np.sum(row * weights)
        portfolio_value *= (1 + period_return)
        portfolio_returns.append(period_return)

        lookback_returns = returns.loc[:date].last('1Y')  # Use past year's data
        if len(lookback_returns) > 30:
            weights = optimize_portfolio(lookback_returns)

    portfolio_returns = pd.Series(portfolio_returns, index=returns.resample(rebalance_frequency).first().index)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    total_return = portfolio_value - 1
    annual_return = (portfolio_value ** (1 / ((end_date - start_date).days / 365))) - 1
    sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()

    return {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Cumulative Returns': cumulative_returns
    }

def plot_portfolio_returns(cumulative_returns):
    plt.figure(figsize=(12, 6))
    cumulative_returns.plot()
    plt.title('Portfolio Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.show()

def plot_portfolio_weights(stocks, weights):
    filtered_stocks = [stock for stock, weight in zip(stocks, weights) if weight > 0]
    filtered_weights = [weight for weight in weights if weight > 0]

    plt.figure(figsize=(12, 6))
    plt.bar(filtered_stocks, filtered_weights)
    plt.title('Portfolio Weights')
    plt.xlabel('Stocks')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.show()



def plot_sharpe_ratio(returns):
    sharpe_ratios = returns.mean() / returns.std()
    plt.figure(figsize=(12, 6))
    sharpe_ratios.plot(kind='bar')
    plt.title('Sharpe Ratio of Each Stock')
    plt.xlabel('Stocks')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.show()


# Main function to execute the program
def main():
    risk_tolerance = input("Enter your risk tolerance (low, medium, high): ").strip().lower()
    tickers = [
    'AAPL',  # Apple Inc.
    'MSFT',  # Microsoft Corporation
    'GOOGL', # Alphabet Inc. (Google)
    'AMZN',  # Amazon.com, Inc.
    'TSLA',  # Tesla, Inc.
    'META',    # Meta Platforms, Inc. (formerly Facebook)
    'NVDA',  # NVIDIA Corporation
    'NFLX',  # Netflix, Inc.
    'INTC',  # Intel Corporation
    'CSCO',  # Cisco Systems, Inc.
    'ADBE',  # Adobe Inc.
    'ORCL',  # Oracle Corporation
    'IBM',   # International Business Machines Corporation
    'PYPL',  # PayPal Holdings, Inc.
    'DIS',   # The Walt Disney Company
    'V',     # Visa Inc.
    'MA',    # Mastercard Incorporated
    'PEP',   # PepsiCo, Inc.
    'KO',    # The Coca-Cola Company
    'WMT'    # Walmart Inc.
]
    
    selected_stocks, optimal_weights = build_and_optimize_portfolio(tickers, risk_tolerance, metrics)
    
    print("\nSuggested Portfolio:")
    for stock, weight in zip(selected_stocks, optimal_weights):
        print(f"{stock}: {weight:.4f}")
    
    # Optional: Run a backtest
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    backtest_results = backtest(selected_stocks, start_date, end_date)
    if backtest_results:
        print("\nBacktest Results:")
        for metric, value in backtest_results.items():
            if metric != 'Cumulative Returns':
                print(f"{metric}: {value:.4f}")

        # Plot cumulative returns
        plot_portfolio_returns(backtest_results['Cumulative Returns'])
        
        # Plot daily returns
        data = yf.download(selected_stocks, start=start_date, end=end_date)
        daily_returns = data['Adj Close'].pct_change().dropna()

        # Plot portfolio weights
        plot_portfolio_weights(selected_stocks, optimal_weights)

        # Plot Sharpe ratio
        plot_sharpe_ratio(daily_returns)

if __name__ == "__main__":
    main()

