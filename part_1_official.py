#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 19:38:24 2024

@authors: Huy Le, Sanketh Udupa, Yucheng Luo, Bhavya Soni

DS2500
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, percentileofscore
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import json
import os
import warnings
from textblob import TextBlob
import requests
import schedule
import time
import warnings
warnings.filterwarnings('ignore')
import os

FEATURES = ['Close', 'MA_5', 'MA_20', 'MA_10', 'RSI', 'MACD']

'''
RUN part_2_official.py TO SEE THE RESULTS
'''

# Function to calculate RSI
def calculate_rsi(data, window=14):
    '''
    Parameters: 
        data: historical data
        window: how many data points to be used for calculating
        
    Returns: 
        rsi
    '''
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
    '''
    Parameters: 
        data: historical data
        window: how many data points to be used for calculating
        
    Returns: 
        macd
    '''
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    macd_hist = macd - signal

    return macd, signal, macd_hist

# Function to calculate SMA
def calculate_sma(data, window):
    '''
    Parameters: 
        data: historical data
        window: how many data points to be used for calculating
        
    Returns: 
        sma
    '''
    return data.rolling(window=window).mean()

# Function to get stock data and calculate metrics
def get_stock_stats(ticker):
    '''
    Parameters: 
        ticker: a stock ticker (symbol)
        
    Returns: 
        return a dictionary {stock stats : stat}
    
        This dictionary will be turned into a df 
            in build_and_optimize_portfolio().
    '''
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
        'Dividend Yield': info.get('dividendYield', 0) * 100 
                            if info.get('dividendYield') else np.nan,
        'Debt to Equity': info.get('debtToEquity', np.nan),
        'ROE': info.get('returnOnEquity', np.nan) * 100 
                            if info.get('returnOnEquity') else np.nan,
        'ROA': info.get('returnOnAssets', np.nan) * 100 
                            if info.get('returnOnAssets') else np.nan,
        'Revenue Growth': info.get('revenueGrowth', np.nan) * 100 
                            if info.get('revenueGrowth') else np.nan,
        'Gross Margins': info.get('grossMargins', np.nan) * 100 
                            if info.get('grossMargins') else np.nan,
        'Free Cash Flow': info.get('freeCashflow', np.nan),
        'Current Ratio': info.get('currentRatio', np.nan),
        'Quick Ratio': info.get('quickRatio', np.nan),
        'Profit Margin': info.get('profitMargins', np.nan) * 100 
                            if info.get('profitMargins') else np.nan,
        'RSI': calculate_rsi(hist['Close'])[-1] if len(hist) > 14 else np.nan,
        'MACD': calculate_macd(hist['Close'])[0][-1] 
                            if len(hist) > 26 else np.nan,
        '50D SMA': calculate_sma(hist['Close'], 50)[-1] 
                            if len(hist) > 50 else np.nan,
        '200D SMA': calculate_sma(hist['Close'], 200)[-1] 
                            if len(hist) > 200 else np.nan,
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
def score_stock(stock_stats, metrics):
    '''
    Parameters:
        row: A row from the stock_stats_df DataFrame
        metrics: A dictionary {stat type : (min_percentile, max_percentile)}
        stock_stats_df: The complete DataFrame containing all stocks' statistics

    Returns:
        score: Score of a stock
    '''
    score = 0
    for metric, (min_percentile, max_percentile) in metrics.items():
        if metric in stock_stats and pd.notna(stock_stats[metric]):
            value = stock_stats[metric]
            if min_percentile <= value <= max_percentile:
                score += 1
    return score

# Function to calculate Sharpe Ratio for optimization
def sharpe_ratio(weights, returns, cov_matrix):
    '''
    Parameters: 
        weights: 
        returns: stock returns
        cov_matrix: 
        
    Returns: 
        sharpe ratio.
    '''
    portfolio_return = np.sum(returns * weights)
    
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    return -portfolio_return / portfolio_volatility

# Function to optimize portfolio weights based on Sharpe Ratio

def optimize_portfolio(stock_data, risk_tolerance):
    """
    Implement portfolio optimization using Mean-Variance Optimization (Markowitz model).

    Parameters:
    stock_data (pd.DataFrame): Historical stock price data
    risk_tolerance (str): 'low', 'medium', or 'high'

    Returns:
    np.array: Optimized weights for each stock
    """
    # Calculate daily returns
    returns = stock_data.pct_change().dropna()

    # Calculate expected returns and covariance matrix
    expected_returns = returns.mean()
    cov_matrix = returns.cov()

    # Define the number of assets
    num_assets = len(stock_data.columns)

    # Define risk tolerance parameters
    if risk_tolerance == 'low':
        target_return = expected_returns.quantile(0.3)
    elif risk_tolerance == 'medium':
        target_return = expected_returns.quantile(0.5)
    else:  # high risk tolerance
        target_return = expected_returns.quantile(0.7)

    # Define the objective function (negative Sharpe Ratio for minimization)
    def objective(weights):
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_volatility

    # Define constraints
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        {'type': 'eq', 'fun': lambda x: np.sum(expected_returns * x) - target_return}  # target return constraint
    )

    # Define bounds for weights
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Set initial weights
    initial_weights = np.array([1/num_assets] * num_assets)

    # Run optimization
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Check if optimization was successful
    if result.success:
        return result.x
    else:
        #print("Optimization failed. Using equal weights.")
        return initial_weights

def build_and_optimize_portfolio(tickers, risk_tolerance, metrics, economic_indicators=None, alternative_data=None):
    # Fetch stock data
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years of data
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    # Get strategy based on risk tolerance
    strategy = investment_strategy(risk_tolerance)

    # Calculate scores for each stock
    stock_scores = {}
    for ticker in tickers:
        stock_stats = get_stock_stats(ticker)
        score = score_stock(stock_stats, metrics)
        stock_scores[ticker] = score

    # Normalize scores to be between 0 and 100
    min_score = min(stock_scores.values())
    max_score = max(stock_scores.values())
    normalized_scores = {ticker: (score - min_score) / (max_score - min_score) * 100
                         for ticker, score in stock_scores.items()}

    # Select top stocks based on normalized scores
    num_stocks = strategy['num_stocks']
    selected_stocks = sorted(normalized_scores, key=normalized_scores.get, reverse=True)[:num_stocks]

    # Optimize weights for selected stocks
    selected_stock_data = stock_data[selected_stocks]
    optimal_weights = optimize_portfolio(selected_stock_data, risk_tolerance)

    # Create final dataframe
    selected_stocks_df = pd.DataFrame({
        'Stock': selected_stocks,
        'Weight': optimal_weights,
        'Score': [normalized_scores[stock] for stock in selected_stocks]
    })

    # Add sentiment scores if available
    if alternative_data is not None and 'Sentiment' in alternative_data.columns:
        selected_stocks_df = selected_stocks_df.merge(
            alternative_data[['Sentiment']],
            left_on='Stock',
            right_index=True,
            how='left'
        )
    else:
        # If sentiment data is not available, add a column of NaN values
        selected_stocks_df['Sentiment'] = np.nan

    return selected_stocks, optimal_weights, selected_stocks_df, strategy

# Function to rebalance the portfolio
def rebalance_portfolio(tickers, risk_tolerance, metrics):
    
    '''
    Parameters: 
        tickers: list of stock tickers
        risk_tolerance: user's input of low, medium or high
        metrics: a dictionary {stat type : {min_percentile, max_percentile}}
    Returns: 
        selected_stocks: list of the tickers of selected stocks
        optimized_weights: numpy nd array
    '''
    
    selected_stocks, optimal_weights = build_and_optimize_portfolio(tickers, 
                                                                    risk_tolerance, 
                                                                    metrics)
    
    print("\nRebalanced Portfolio:")
    
    for stock, weight in zip(selected_stocks, optimal_weights):
        
        print(f"{stock}: {weight:.4f}")
        
    return selected_stocks, optimal_weights

# Function to backtest the portfolio
def backtest(stocks, risk_tolerance, 
             start_date, end_date, rebalance_frequency='M'):
    '''

    Parameters
    ----------
    stocks : list of selected tock tickers.
    start_date : start date in the form 'yyyy-mm-dd'.
    end_date : end date in the form 'yyyy-mm-dd'.
    
    rebalance_frequency : str, default is 'M'.

    Returns
    -------
    dict
        {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Cumulative Returns': cumulative_returns
        }

    '''
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
            weights = optimize_portfolio(lookback_returns, risk_tolerance)

    portfolio_returns = pd.Series(
                    portfolio_returns, 
                    index=returns.resample(rebalance_frequency).first().index)
    
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
    plt.savefig('porfolio_returns.jpg', edgecolor='black', dpi=400, 
                facecolor='orange', transparent=True)
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
    plt.savefig('porfolio_weights.jpg', edgecolor='black', dpi=400, 
                facecolor='orange', transparent=True)
    plt.show()



def plot_sharpe_ratio(returns):
    
    sharpe_ratios = returns.mean() / returns.std()
    plt.figure(figsize=(12, 6))
    sharpe_ratios.plot(kind='bar')
    plt.title('Sharpe Ratio of Each Stock')
    plt.xlabel('Stocks')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.savefig('sharpe_ratio.jpg', edgecolor='black', dpi=400, 
                facecolor='orange', transparent=True)
    plt.show()

def display_portfolio_stats(selected_stocks, optimal_weights, risk_tolerance):
    
    '''
    Parameters: 
        selected_stocks: list of the tickers of selected stocks
        optimized_weights: numpy nd array
    Returns: 
        Nothing, but prints text, and displays graphs
    '''
    
    print("\nSuggested Portfolio:")
    for stock, weight in zip(selected_stocks, optimal_weights):
        print(f"{stock}: {weight:.4f}")
    
    # Optional: Run a backtest
    # start in 2021 to avoid Covid
    start_date = '2021-01-01'
    end_date = '2024-01-01'
    backtest_results = backtest(selected_stocks, risk_tolerance, 
                                start_date, end_date)
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


def get_sentiment(ticker):
    """
    Fetch and analyze the sentiment of news headlines related to a specific stock ticker.

    Parameters:
    ticker: str
        The stock ticker symbol for which to retrieve and analyze news headlines.

    Returns:
    float
        The average sentiment score of the news headlines. 
        The sentiment score ranges from -1 (very negative) to 1 (very positive).
        If no valid headlines are found or an error occurs, the function returns 0.

    Detailed Steps:
    --------------
    1. **Fetch News Headlines:**
       - The function constructs a URL using the NewsAPI endpoint, 
       incorporating the stock ticker symbol and the API key.
       - An HTTP GET request is made to fetch news articles related 
       to the specified ticker.

    2. **Handle API Response:**
       - The function checks if the request was successful 
       by calling `response.raise_for_status()`. 
       If the request fails, an exception is raised and caught.
       - The JSON response is parsed to extract news articles. 
       If the 'articles' key is missing from the response, 
       a message is logged, and the function returns a sentiment score of 0.

    3. **Sentiment Analysis:**
       - The function iterates over the first 10 articles in the 'articles' list.
       - For each article, it checks if the 'title' key is present. 
       If the title exists, it is analyzed using `TextBlob` 
       to determine its sentiment polarity.
       - The sentiment polarity scores are collected in the `sentiment_scores` list.
       - If an article does not contain a 'title', a message is logged 
       indicating the missing data.

    4. **Calculate and Return Average Sentiment:**
       - The average sentiment score is calculated 
       by summing the `sentiment_scores` and dividing by the number of scores.
       - If no valid sentiment scores are available, the function returns 0.

    5. **Error Handling:**
       - If any network-related issues occur (e.g., connection errors), 
       they are caught by `requests.RequestException`, 
       and a sentiment score of 0 is returned.
       - Any other unexpected errors are caught by a general exception handler, 
       logged, and result in a sentiment score of 0.
    """
    
    '''
    visit newsapi.org, create an account and get the API Key. 
    Then replace the api key in the link below.
    '''
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=92851de6996f41d6b874c8a4b60ceea5"

    try:
        response = requests.get(url)
        response.raise_for_status()
        news = response.json()

        if 'articles' not in news:
            print(f"API response for {ticker} does not contain 'articles'. \
Full response: {news}")
            return 0

        sentiment_scores = []
        for article in news['articles'][:10]:
            if 'title' in article:
                analysis = TextBlob(article['title'])
                sentiment_scores.append(analysis.sentiment.polarity)
            else:
                print(f"Article for {ticker} does not contain 'title'. \
Article data: {article}")

        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

    except requests.RequestException as e:
        print(f"Error fetching news for {ticker}: {e}")
        return 0
    except Exception as e:
        print(f"Unexpected error processing sentiment for {ticker}: {e}")
        return 0

def update_current_portfolio(new_portfolio):
    """
    Update the current portfolio by saving the provided portfolio data to a JSON file.

    Parameters:
    new_portfolio: dict
        A dictionary containing the updated portfolio information. 
       This could include stock tickers, quantities, weights, or 
       any other relevant data.

    Detailed Steps:
    --------------
    1. **Specify the Portfolio File:**
       - The function defines the file name `portfolio_file` as 
       'current_portfolio.json'. This is the file where the portfolio data will be saved.

    2. **Write the Portfolio to the JSON File:**
       - The function opens the specified JSON file in write mode (`'w'`).
       - The `json.dump` function is used to serialize the 
       `new_portfolio` dictionary into JSON format and write it to the file.
       - If the file already exists, it will be overwritten with 
       the new portfolio data. If it doesn't exist, a new file will be created.

    3. **File Handling:**
       - The file is automatically closed when the `with` block is exited, 
       ensuring that all data is properly saved and the file is not left open.

    Notes:
    -----
    - This function assumes that the `new_portfolio` dictionary is correctly 
    formatted and ready to be serialized into JSON.
    - If any errors occur during the file writing process 
    (e.g., file permission issues), an exception will be raised, w
    hich is not handled within this function.
      It is recommended to handle such exceptions where this function 
      is called to ensure robustness.
    """
    portfolio_file = 'current_portfolio.json'
    with open(portfolio_file, 'w') as f:
        json.dump(new_portfolio, f)


def add_features(df, days):
    '''
    Parameters:
        df: a stock dataframe with these columns
            'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'

        days: user's input of how many days to predict
    Returns:
        df: with these columns:
            'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'MA_5', 'MA_20', 'MA_10', 'shifted'

        projection_features: a df with these columns:
            'Close', 'MA_5', 'MA_20', 'MA_10'
    '''

    df['MA_5'] = calculate_sma(df['Close'], 5)
    df['MA_20'] = calculate_sma(df['Close'], 20)
    df['MA_10'] = calculate_sma(df['Close'], 10)
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], _, _ = calculate_macd(df['Close'])  # 修改这里
    df['shifted'] = df['Close'].shift(-days)

    df = df.dropna()

    projection_features = df[['Close', 'MA_5', 'MA_20', 'MA_10']].tail(days)

    return df, projection_features

def predict_stock_price(ticker, days):
    data = yf.download(ticker, start='2020-01-01', end=pd.Timestamp.today())
    data = add_features(data, days)
    X = data[FEATURES]
    y = data['shifted']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, shuffle=False)
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    predictions = knn_model.predict(X_test)
    return predictions[-1]

def get_current_portfolio():
    # This function should return a dictionary with the stock code as the key and the position ratio as the value
    # In practice, this function should connect to your brokerage account or database to get real-time data

    # For demonstration purposes, we will use a simulated portfolio file
    portfolio_file = 'current_portfolio.json'

    if os.path.exists(portfolio_file):
        with open(portfolio_file, 'r') as f:
            return json.load(f)
    else:
        # If the file does not exist, return an empty portfolio
        return {}

def execute_trades(current_portfolio, target_portfolio):
    for stock, target_weight in target_portfolio.items():
        current_weight = current_portfolio.get(stock, 0)
        if target_weight > current_weight:
            print(f"Buy {stock}: {target_weight - current_weight:.4f}")
            # Add the actual buying logic here
        elif target_weight < current_weight:
            print(f"Sell {stock}: {current_weight - target_weight:.4f}")
            # Add the actual selling logic here

    # Update the current portfolio
    update_current_portfolio(target_portfolio)

def automated_trading(tickers, risk_tolerance, metrics, strategy, 
                      economic_indicators, alternative_data):
    def job():
        try:
            print(f"Rebalancing portfolio at {datetime.now()}")
            selected_stocks, optimal_weights = build_and_optimize_portfolio(tickers, risk_tolerance, metrics)
            current_portfolio = get_current_portfolio()
            target_portfolio = dict(zip(selected_stocks, optimal_weights))
            execute_trades(current_portfolio, target_portfolio)
        except Exception as e:
            print(f"Error during automated trading: {e}")

    schedule.every().day.at("16:00").do(job)

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Automated trading stopped by user.")



def investment_strategy(risk_tolerance):
    """
    Define investment strategy based on user's risk tolerance.

    Parameters:
    risk_tolerance (str): User's risk tolerance level ('low', 'medium', or 'high')

    Returns:
    dict: A dictionary containing strategy parameters
    """
    if risk_tolerance == 'low':
        return {
            'num_stocks': 5,
            'max_weight': 0.3,
            'min_weight': 0.05,
            'rebalance_frequency': 'Q',  # Quarterly
            'stop_loss': 0.05,
            'take_profit': 0.1
        }
    elif risk_tolerance == 'medium':
        return {
            'num_stocks': 10,
            'max_weight': 0.2,
            'min_weight': 0.03,
            'rebalance_frequency': 'M',  # Monthly
            'stop_loss': 0.1,
            'take_profit': 0.15
        }
    else:  # high risk tolerance
        return {
            'num_stocks': 15,
            'max_weight': 0.15,
            'min_weight': 0.02,
            'rebalance_frequency': 'W',  # Weekly
            'stop_loss': 0.15,
            'take_profit': 0.2
        }


def risk_parity_strategy(stock_data, risk_tolerance):
    """
    Implement a simple risk parity strategy.

    This strategy allocates capital to assets based on their risk contribution,
    aiming to equalize risk across all assets in the portfolio.
    """
    returns = stock_data.pct_change().dropna()
    cov_matrix = returns.cov()

    def risk_budget_objective(weights, args):
        cov_matrix = args[0]
        assets_risk = np.sqrt(np.diagonal(cov_matrix))
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        risk_contribution = weights * (np.dot(cov_matrix, weights) / portfolio_risk)
        return np.sum((risk_contribution - portfolio_risk / len(weights)) ** 2)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(stock_data.columns)))

    result = minimize(risk_budget_objective,
                      [1 / len(stock_data.columns)] * len(stock_data.columns),
                      args=[cov_matrix],
                      method='SLSQP',
                      constraints=constraints,
                      bounds=bounds)

    return pd.Series(result.x, index=stock_data.columns)


def economic_indicators_strategy(stock_data, risk_tolerance):
    economic_data = get_economic_indicators()
    returns = stock_data.pct_change().dropna()
    correlations = returns.corrwith(economic_data)
    weights = (correlations + 1) / (correlations + 1).sum()
    return weights


def multi_factor_strategy(stock_data, factors):
    """
    Implement a multi-factor strategy.

    This strategy combines multiple factors to determine stock weights.
    """
    factor_scores = pd.DataFrame(index=stock_data.columns)

    for factor, data in factors.items():
        factor_scores[factor] = data
    factors = {
        'momentum': stock_data.pct_change(periods=20),
        'volatility': stock_data.pct_change().rolling(window=20).std(),
        'value': 1 / stock_data  # This is a simple price-to-book ratio proxy
    }
    return multi_factor_implementation(stock_data, factors)

    # Normalize factor scores
    factor_scores = (factor_scores - factor_scores.mean()) / factor_scores.std()

    # Combine factors (equal-weighted for simplicity)
    combined_score = factor_scores.mean(axis=1)
    weights = combined_score / combined_score.sum()
    return weights

def multi_factor_implementation(stock_data, factors):
    factor_scores = pd.DataFrame(index=stock_data.columns)
    for factor, data in factors.items():
        factor_scores[factor] = data.iloc[-1]  # Use the most recent data point
    factor_scores = (factor_scores - factor_scores.mean()) / factor_scores.std()
    combined_score = factor_scores.mean(axis=1)
    weights = combined_score / combined_score.sum()
    return weights
def stop_loss_strategy(stock_data, risk_tolerance):
    strategy = investment_strategy(risk_tolerance)
    stop_loss_threshold = strategy['stop_loss']
    """
    Implement a simple stop-loss strategy.

    This strategy adjusts weights based on whether stocks have hit the stop-loss threshold.
    """
    returns = stock_data.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod()
    drawdowns = (cumulative_returns / cumulative_returns.cummax() - 1)

    weights = pd.Series(1 / len(stock_data.columns), index=stock_data.columns)
    for stock in drawdowns.columns:
        if drawdowns[stock].min() <= -stop_loss_threshold:
            weights[stock] = 0

    weights = weights / weights.sum()  # Renormalize
    return weights


def ml_prediction_strategy(stock_data,risk_tolerance):
    """
    Implement a machine learning prediction strategy.

    This strategy uses a Random Forest model to predict future returns and allocate accordingly.
    """
    returns = stock_data.pct_change().dropna()
    predictions = {}
    for stock in returns.columns:
        X = np.arange(len(returns)).reshape(-1, 1)
        y = returns[stock].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions[stock] = model.predict(X_test)[-1]
    weights = pd.Series(predictions)
    weights = weights.clip(lower=0)
    weights = weights / weights.sum()
    return weights

def alternative_data_strategy(stock_data, risk_tolerance):
    """
    Implement a strategy using alternative data.

    This strategy adjusts weights based on alternative data signals.
    """
    # Assuming alternative_data is a DataFrame with the same index as stock_data
    # and columns representing different alternative data signals

    # Normalize alternative data
    alternative_data = get_alternative_data(stock_data.columns)
    normalized_data = (alternative_data - alternative_data.mean()) / alternative_data.std()
    combined_signal = normalized_data.mean(axis=1)
    weights = combined_signal / combined_signal.sum()
    return weights


def get_economic_indicators():
    """
    Fetch and calculate real economic indicators from the FRED API.
    
    Create an account at api.stlouisfed.org and get the API Key.
    
    Returns:
    pd.Series: A series of economic indicators
    """
    api_key = "39a3ef00eb52d3222ac774b35fd92916"
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    indicators = {
        'GDP_Growth': 'GDP',
        'Inflation_Rate': 'CPIAUCSL',
        'Unemployment_Rate': 'UNRATE',
        'Interest_Rate': 'DFF'
    }

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    data = {}
    for name, series_id in indicators.items():
        url = f"{base_url}?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={start_date}&observation_end={end_date}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            observations = response.json()['observations']

            if name in ['GDP_Growth', 'Inflation_Rate']:
                # Calculate annual growth rate
                oldest = float(observations[0]['value'])
                newest = float(observations[-1]['value'])
                growth_rate = (newest - oldest) / oldest * 100
                data[name] = growth_rate
            elif name == 'Unemployment_Rate':
                # Use the most recent value
                data[name] = float(observations[-1]['value'])
            elif name == 'Interest_Rate':
                # Calculate average over the period
                values = [float(obs['value']) for obs in observations]
                data[name] = sum(values) / len(values)
        except requests.RequestException as e:
            print(f"Failed to fetch {name}: {e}")
            data[name] = np.nan

    return pd.Series(data)

def get_alternative_data(tickers):
    """
    Fetch alternative data for given tickers.

    Parameters:
    tickers (list): List of stock tickers

    Returns:
    pd.DataFrame: A dataframe of alternative data for each ticker
    """
    alt_data = pd.DataFrame(index=tickers)

    # Fetch sentiment using existing get_sentiment function
    alt_data['Sentiment'] = [get_sentiment(ticker) for ticker in tickers]

    # Fetch trading volume as a proxy for investor interest
    try:
        volume_data = yf.download(tickers, period="1mo")['Volume']
        alt_data['Relative_Volume'] = volume_data.mean() / volume_data.mean().mean()
    except Exception as e:
        print(f"Error fetching volume data: {e}")
        alt_data['Relative_Volume'] = np.random.randn(len(tickers))

    return alt_data