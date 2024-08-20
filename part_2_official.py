#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 19:38:24 2024

@authors: Huy Le, Sanketh Udupa, Yucheng Luo, Bhavya Soni

DS2500
"""

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import date, timedelta, datetime
import os
import warnings
warnings.filterwarnings('ignore')

import part_1_official


FEATURES = ['Close', 'MA_5', 'MA_20', 'MA_10']
TICKER = 'AAPL'


def moving_avg(lst, window = 2):
    '''
    Takes in a lst of numbers and window.
    Window is the number of data points used in the moving avg process.
    
    Returns a list of averages.
    '''
    avgs = []
    
    for extra in range(window - 1):
        avgs.append(np.nan)
    
    for pos in range(len(lst) - window + 1):
        
        temp_avg = 0
        
        for idx in range(window):
            
            temp_avg += lst[pos + idx]
            
        avgs.append(temp_avg / window)
        
    return avgs


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
    
    df['MA_5'] = moving_avg(df['Close'], 5)
    df['MA_20'] = moving_avg(df['Close'], 20)
    df['MA_10'] = moving_avg(df['Close'], 10)
    
    df['shifted'] = df['Close'].shift(-days)
    
    projection_features = df.iloc[-days:][['Close', 'MA_5', 'MA_20', 'MA_10']]
    
    df = df.dropna()
    
    return df, projection_features

def predict(df, label, projection=[]):
    
    '''
    Parameters: 
        df: a stock dataframe with these columns: 
            'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
            'MA_5', 'MA_20', 'MA_10', 'shifted'
            
        label: 'historic' or 'future'
        
        projection: a lst version of projection_features
        
    Returns: 
        predictions: a lst
    '''
    
    X = df[FEATURES]
    y = df['shifted']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, 
                                                        shuffle=False)
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    
    
    if label == "historic":
        predictions = knn_model.predict(df[FEATURES])
    if label == "future":
        predictions = knn_model.predict(projection)
    
    return predictions


def calc_new_dates(old_dates, days):
    
    '''
    Parameters: 
        old_dates: a lst of dates from past data
        days: user's input of how many days to predict
        
    Returns: 
        new_dates: lst of dates in the future
    '''
    
    days_ct = days
    
    #old_dates = lst.copy()
    new_dates = []
    
    for old_date in old_dates:
        
        days_ct = days
        
        while days_ct > 0:
            
            old_date = old_date + timedelta(days=1)
            
            if old_date.weekday() >= 5:
                continue
            
            days_ct -= 1

        new_dates.append(old_date)

    return new_dates

def historic_plot(df, past_predictions, days, selected_stock):
    
    '''
    Parameters: 
        df: a df with these columns: 
            'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
            'MA_5', 'MA_20', 'MA_10', 'shifted'
            
        past_predictions: predictions of past data
        
        days: user's input of how many days to predict
        
        selected_stock: a stock in the selected_stocks list
        
    Returns: 
        Nothing, but a line plot
    '''

    new_dates = calc_new_dates(list(df.index), days)    

    plt.figure(figsize=(14,7))
    plt.plot(new_dates, df['shifted'].values, label="Actual Values")
    plt.plot(new_dates, past_predictions, label="Predicted Values")
    plt.title(f"Historical Stock Price Data for {selected_stock} \
({min(df.index).strftime('%Y/%m/%d')} - {max(df.index).strftime('%Y/%m/%d')})", 
                fontsize=22)
        
    plt.legend(fontsize=15)
    plt.xlabel("Dates", fontsize=20)
    plt.ylabel("Stock Price", fontsize=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()
    


def focused_plot(df, projection_features, days_ct, projected, 
                 selected_stock):
    
    '''
    Parameters: 
        df: a df with these columns: 
            'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
            'MA_5', 'MA_20', 'MA_10', 'shifted'
        
        projection_features: a df with these columns: 
            'Close', 'MA_5', 'MA_20', 'MA_10'
        
        days_ct: user's input of how many days to predict
            
        projected: FUTURE predictions from predict()
        
        selected_stock: a stock in the selected_stocks list
        
        
    Returns: 
        Nothing, 
        but plots historic prices, predicted prices, and weekend
    '''
    
    new_dates = calc_new_dates(list(df.index), days_ct)
    projected_dates = calc_new_dates(list(projection_features.index), days_ct)
    
    start_date = new_dates[-(4*days_ct):]

    weekends = [(min(start_date) + timedelta(days=days)) for days 
                in range((max(projected_dates) - min(start_date)).days + 1) 
                if ((min(start_date) + timedelta(days=days)).weekday() >= 5)]
    

    newdates = pd.DataFrame({'dates': list(new_dates[-(4*days_ct):]) 
                             + list(projected_dates), 
                             'prices': list(df['shifted'].values[-(4*days_ct):]) 
                             + list(projected)})

    
    weekend_prices1 = {}

    for weekend in weekends:
        
        weekday = weekend
        
        while not weekday in new_dates:
            
            weekday -= timedelta(days=1)
            
        weekend_prices1[weekend] = float(newdates.loc[newdates['dates'] == weekday, 
                                                      'prices'])
        
        
    
    # for key, value in weekend_prices1.items():
        
    #     new_row = pd.DataFrame([[key, value]], columns=['dates', 'prices'])
    #     newdates = pd.concat([newdates, new_row], ignore_index=True)

    
    newdates = newdates.sort_values(by="dates")
    newdates = newdates.reset_index(drop=True)

    formatted_dates = [date.strftime('%m/%d') for date in newdates['dates']]

    
    newdates['color'] = 'black'
    for idx in newdates.index:
        
        if newdates['dates'][idx] in start_date:
            newdates['color'][idx] = 'green'
        elif newdates['dates'][idx] in weekends:
            newdates['color'][idx] = 'black'
        else:
            newdates['color'][idx] = 'blue'
    
    plt.figure(figsize=(14, 7), dpi=400)
    
    dates = newdates['dates']
    prices = newdates['prices']
    colors = newdates['color']
    
    for date1,date2, price1,price2, color in zip(dates, dates[1:], prices, 
                                                 prices[1:], colors):
        
        plt.plot([date1, date2], [price1, price2], color=color, linewidth=4, 
                 marker='o', markersize=8)
        
    predict_patch = mpatches.Patch(color='blue', label='Predicted Prices')
    history_patch = mpatches.Patch(color='green', label='Historic Prices')
    # weekend_patch = mpatches.Patch(color='black', label='Weekend (No Trading)')
    
    plt.legend(handles=[history_patch, predict_patch], 
               loc='upper left', framealpha=1, frameon=True, fontsize=15)
    
    plt.title(f"{selected_stock} Stock Price Data with {days_ct} \
Day Predictions (2024)", 
              fontsize=22)
    plt.xlabel("Dates", fontsize=20)
    plt.ylabel("Stock Price", fontsize=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()
    
def plot_porfolio(days, selected_stocks):
    
    '''
    Deals with EACH of the SELECTED stocks:
        -returns and plots predictions of PAST values
        -returns and plots predictions of FUTURE values
        -
    
    '''
    
    for selected_stock in selected_stocks: 
        
        data = yf.download(selected_stock, start='2020-01-01', 
                           end=date.today() + timedelta(days=1))
        '''
        by now, data has these columns: 
            'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
        '''
        
        data = pd.DataFrame(data)
        '''
        It's more readable to create a df of ONLY ONE stock,
        then add these columns: 
            'MA_5', 'MA_20', 'MA_10', 'shifted'
            
        projection_features contain 'Close', 'MA_5', 'MA_20', 'MA_10' columns
        '''
        data, projection_features = add_features(data, days)
    
        
        past_predictions = predict(data, "historic")
        
        projected = predict(data, "future", projection_features)
        
        historic_plot(data, past_predictions, days, selected_stock)
        
        focused_plot(data, projection_features, days, 
                     projected, selected_stock)
        
        new_dates = calc_new_dates(projection_features.index, days)
        
        print('\n')
        print(f"The predicted stock price for {selected_stock} for the next {days} " \
              "business days is: ")
        print('\n')
            
        for idx in range(len(new_dates)):
            print(f"{new_dates[idx].strftime('%Y-%m-%d')}: {round(projected[idx], 4)}")
        
        #print(data.columns)


def main():
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

    risk_tolerance = input("Enter your risk tolerance (low, medium, high): ").strip().lower()
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'INTC', 'CSCO',
        'ADBE', 'ORCL', 'IBM', 'PYPL', 'DIS', 'V', 'MA', 'PEP', 'KO', 'WMT'
    ]

    print("Welcome to the White Rock Capital!")

    # Initialize the portfolio file (if it doesn't exist)
    if not os.path.exists('current_portfolio.json'):
        initial_portfolio = {}
        part_1_official.update_current_portfolio(initial_portfolio)

    print("\nAnalyzing stocks...")

    # Get economic indicators
    economic_indicators = part_1_official.get_economic_indicators()
    print("\nCurrent Economic Indicators:")
    print(economic_indicators)

    # Get alternative data
    alternative_data = part_1_official.get_alternative_data(tickers)
    print("\nAlternative Data Summary:")
    print(alternative_data.describe())

    # Build and optimize portfolio
    selected_stocks, optimal_weights,\
        selected_stocks_df, strategy = part_1_official.build_and_optimize_portfolio(
                                        tickers, risk_tolerance, metrics, 
                                        economic_indicators, alternative_data)

    # Display investment strategy
    print("\nSelected Investment Strategy:")
    for key, value in strategy.items():
        print(f"{key}: {value}")

    # Display portfolio composition
    print("\nOptimized Portfolio Composition:")
    for index, row in selected_stocks_df.iterrows():
        print(f"{row['Stock']}: Weight = {row['Weight']:.4f}, Score = {row['Score']:.2f}, Sentiment = {row['Sentiment']:.4f}")

    # Calculate and display average score and sentiment
    avg_score = selected_stocks_df['Score'].mean()
    print(f"\nAverage Portfolio Score: {avg_score:.2f}")

    if 'Sentiment' in selected_stocks_df.columns and not selected_stocks_df['Sentiment'].isnull().all():
        avg_sentiment = selected_stocks_df['Sentiment'].mean()
        print(f"Average Sentiment Score: {avg_sentiment:.4f}")
    else:
        print("Sentiment data not available.")

    # Ask the user if they want to start automated trading
    auto_trade = input("\nDo you want to start automated trading? (yes/no): ").strip().lower()
    if auto_trade == 'yes':
        print("Starting automated trading... Press Stop Kernel to stop.")
        
        part_1_official.automated_trading(tickers, risk_tolerance, metrics, strategy, 
                                     economic_indicators, alternative_data)
    else:
        print("Automated trading not started. You can manually rebalance your portfolio.")

    days = int(input("How many days do you want to predict: "))

    plot_porfolio(days, selected_stocks)

    part_1_official.display_portfolio_stats(selected_stocks, optimal_weights, 
                                            risk_tolerance)
    
    print("\nThank you for using the Advanced Portfolio Optimization System!")

if __name__ == "__main__":
    main()
