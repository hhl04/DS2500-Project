**THIS PROJECT AND ITS RESULTS DO NOT CONSTITUTE A FINANCIAL ADVICE**



**DEMO 1**

https://github.com/user-attachments/assets/425a65d2-b748-4afc-b4b1-9ba2a0ebadc4


**DEMO 2**

https://github.com/user-attachments/assets/1c7dbf0d-bd7d-43eb-94bd-aa3c41c2bd6c



PROBLEM STATEMENT AND BACKGROUND

  Everyone wants to make more money; it’s the reality. One of the more prominent ways to increase your money outside of a regular salary is through investing. 
  Investing is so popular because with the right investments, your money will make more for you by itself. 
  The main drawback which scares many people from investing is the riskiness of the stock and the potential to lose everything you’ve invested in. 
  There are many types of investing, like through real estate or bonds. We chose to further investigate investing in stocks. 
  More specifically, we wanted to create and optimize a stock portfolio which would recommend certain stocks and the size of the investment for that stock, 
    considering the riskiness of the stock and the appetite of the consumer. 
  There are many layers to stock investing, which becomes exponentially more complex the deeper you go. 
  We wanted to create a basic portfolio, which considered three main things: company strength or how profitable the company is, 
    growth potential or if the company has the capacity to grow, and risk level or how stable the company is. 

INTRODUCTION TO OUR DATA

  Our main data source is share price data from various public companies from January 2020 to present. 
  
  To retrieve all this data from the various companies, our project heavily depends on the yfinance library which eases the data retrieval. 
  With this library, we can download data of a single stock or of multiple stocks simultaneously from the Yahoo Finance website. 
  After downloading data of a single stock and converting it into a data frame, the data frame will have 'Open', 'High', 'Low', 'Close', 'Adj Close', and 'Volume' columns 
    and the index column contains ‘yyyy-mm-dd’ values. 
  When importing data of multiple stocks, each column in the data frame contains the closing values of a stock, and the index column contains ‘yyyy-mm-dd’ values. 
  With yfinance, we can also specify the start and end dates. The yfinance library also displays a helpful progress bar whenever users retrieve stock data using this library. 
  
  There aren’t any privacy and ethical concerns with collecting this data, as it is based on publicly available information from companies that have gone public. 
  Yahoo Finance is widely recognized for being a reliable source and is directly sourced by these public companies. 
  The only identifiable information in the data is about the companies, which is not a concern as the companies are public. 
  Furthermore, there is no specific information on people within each company. There is very minimal bias in the data itself, and there is nothing omitted in the data, 
    especially since Yahoo Finance updates its data in real time. 
  Beside using yfinance to retrieve data, we also used alternative data sources such as sentiment analysis and economic indicators. 
  We used NewsAPI for sentiment analysis and the FRED® API for accessing current economic indicators.
  NewsAPI allows us to gather the latest news headlines related to specific stocks for sentiment analysis, so we can get real-time articles from various news sources. 
  And we can specify the stock ticker with relevant articles. This data can be used to get the market sentiment for the stock we chose and help us to know the market sentiment, 
    which helps us to make investment decisions. 
  The FRED® API provides reliable and comprehensive economic data from the US Federal Reserve System. 
  Indicators such as GDP growth rate, inflation rate, unemployment rate, and interest rates help us evaluate the economic environment, 
    so we have a macroscopic perspective to our investment strategies.	
  The NewsAPI provides the news content publicly, and the FRED® API provides officially published economic data. Neither source contains personally identifiable information or sensitive data.
  
DATA SCIENCE APPROACHES

  Obviously, an essential thing to creating a stock portfolio is being able to predict future stock prices. To do so, we used the KNN machine learning algorithm. 
  We originally considered the Linear Regression Model, but soon realized that stock market data is not linear and instead has many fluctuations. 
  On the other hand, KNN can handle multiple numerical features, and capture complex and nonlinear patterns in the data. For the features of the machine learning algorithm, 
    we decided to use moving averages. We used this for several reasons, one of them being to smooth out the data, as it eliminates any drastic outliers 
    by averaging out the prices for a given window. Another reason is it effectively captures recent and long-term trends when we use moving averages of windows 5 and 20 respectively. 
  We also included a moving average of 10 to capture any intermediate trends. Hence, we used moving averages with window 5, 10 and 20 of the closing prices for each stock as features for KNN. 
  For the label of the model, we used the closing price of the stock. A sample of the data frame is shown below with the AAPL stock. 
  For ease of viewing, we’ve only displayed the moving averages with windows of 5 and 20.
  
  <img width="460" alt="Ảnh màn hình 2024-08-20 lúc 15 51 05" src="https://github.com/user-attachments/assets/dc4d9a41-d6b4-4a50-9de2-9cf97625ed85">

  An essential part of the prediction was to shift the closing price back, so the moving average features are trained to predict future closing prices. 
  For example, if we wanted to predict share prices one day ahead, we would shift the closing price one day back. 
  As can be seen in the diagram above, shifting the cell creates a blank cell (bottom right). 
  That cell is where the prediction of the next day would go. The dates would also be shifted one day back to match the closing prices. 
  Therefore, the moving averages of $219.468 and $224.724 would be trained for Aug. 5’s closing price of $209.27. 
  Since this table was made on Aug 5, that is the last closing price we have in the data frame. 
  For Aug. 6, it would use the moving average of $217.674 and $223.797 to make the prediction.

  To create the stock portfolio, we used the Markowitz Model. 
  This portfolio theory uses historical data to calculate the expected return and covariance matrix, and then finds the optimal weight to maximize the sharpe ratio 
    in order to optimize asset allocation by balancing risk and return. 
  Stock weights are chosen using various metrics to guide decision-making. 
  The code processes stocks based on a list of ticker symbols provided in the tickers = [] list within the main function. 
  It retrieves historical data for each ticker symbol, such as stock prices and trading volumes, by accessing Yahoo Finance. 
  This data is then compiled into a dataframe, which serves as a structured repository for the information related to each company.
  
  To enhance the dataset, the code applies various functions to add important financial metrics and indicators to the dataframe. 
  These functions include sharpe_ratio, which calculates the risk-adjusted return of the stocks; 
    calculate_sma, which computes the Simple Moving Average to smooth out price data over a specified period; 
    and calculate_rsi, which measures the speed and change of price movements to identify overbought or oversold conditions. 
  By incorporating these features/metrics, the data frame becomes more robust, allowing for more in-depth analysis and informed decision-making in tasks 
    such as portfolio optimization and stock selection.
  The metrics derived from these indicators are then used to inform the optimization process. 
  Specifically, the code employs an optimization algorithm, such as the minimize function from the scipy.optimize library, to determine the optimal portfolio weights. 
  The optimization algorithm uses the metrics as inputs to balance the trade-off between risk and return. 
  For example, an investor might aim to maximize the portfolio's expected return while keeping the overall volatility (a measure of risk) within acceptable limits.
  
  We also use Natural Language Processing and sentiment analysis. Basically we use the TextBlob library for sentiment analysis of news headlines. 
  This technique helps us convert text data into a numerical sentiment score that helps us to quantify the market sentiment.

RESULTS AND CONCLUSION

  **Future Stock Price Prediction**
  
  The first graph our program creates is a historical plot with data from January 2020 to August 2024, which is shown below for a sample stock AAPL.
  
  <img width="460" alt="Ảnh màn hình 2024-08-20 lúc 15 52 48" src="https://github.com/user-attachments/assets/3c525dd9-513a-4b61-bdd4-d4fe4d41172a">

  This graph is the result of our KNN machine learning algorithm given some training values, and testing them on all trading days from January 2020 to August 2024. 
  The blue are the actual closing prices of the stock, and the orange is the predicted values. 
  As evidenced by the graph, the machine learning model did a decent job at predicting the prices. 
  The RMSE value was a bit on the higher side of 2.1, but this is a reasonable score, given the size of the sample, which was about 1200 values. 
  The RMSE value does get higher, the more dates you ask the program to predict. This is natural, as predictions are weaker in any sense, the farther out it is. 
  In our program, we create historical plots for many stocks to use in our analysis for our portfolio. 
  Next, the program creates the plots with the actual predictions. This plot is shown below for AAPL and was asked to predict 8 days.
  
  <img width="713" alt="Ảnh màn hình 2024-08-20 lúc 15 53 21" src="https://github.com/user-attachments/assets/26e74e0d-5a56-4245-9513-11fdf95cbea9">
  
  The blue is the next predicted days from when this plot was created. 
  The green is the historic days, which was important to include to provide context to where the predictions are coming from. 
  Simultaneous to this plot being produced, there will also be a text output, giving the exact predictions of the stocks for the predicted number of days. 
  One of the difficult parts of making predictions like these was to filter out the weekends, which you can see was removed for Aug 17 and 18 because there were weekends. 
  No trading happens on weekends, so there is no reason to include that in the prediction.

  **Optimizing Weights**
  
  This following chart shows the strategy for high risk tolerance users. 
  In our investment optimization strategy, we use the Markowitz model in the optimiz_portfolio function to optimize the portfolio by maximizing the Sharpe ratio. 
  The goal is to achieve the highest return at a given risk level or minimize risk at a given return level. Therefore, the allocation of stock weights reflects 
    each stock’s contribution to the Sharpe ratio. 
  In the results of the Optimized Portfolio Composition, we can see that KO(coca-cola) takes the largest weight where V(VISA) takes the least weight, 
    because our weight allocation model is heavily dependent on historical data, and the time range we selected is the past two years, 
    during which KO has shown very good volatility, which will be interpreted as as good stock in our risk management focused optimize_portfolio function. 
  Our current stock pool sample is too small, and if we have more choices, it will help us diversify the portfolio and increase its diversity. 
  That’s why in the presentation, we believe that the current risk of this portfolio comes from the fact that we may be more dependent on individual stocks 
    and thus increase the non systematic risk. 
  Because the risks we currently face are not primarily from the risk of individual companies, but from the risk of dependence resulting from a weighting bias towards a particular stock. 
  However, this risk can be solved manually, as we have set the rebalance frequency in our selected investment strategy module to weekly. 
  In other words, this risk can be avoided by monitoring the market and stock conditions indices on a weekly basis and adjusting our portfolio accordingly.

  <img width="319" alt="Ảnh màn hình 2024-08-20 lúc 15 55 03" src="https://github.com/user-attachments/assets/dc227cc8-2ebf-4cb8-9ec2-a39fc5932060">

  According to the result of optimized portfolio composition, the sentiment indexes don't affect our portfolio since we didn’t consider this factor in the weight calculation part. 
  And currently we focused more on financial data analysis, so we won’t put sentiment in our weight calculation, but we will refer to it if we need to adjust our portfolio strategy.
  
  The final output of our project includes a graph shown below that visualizes the optimized stock weights based on the investor's risk tolerance. 
  The portfolio reflects an optimal distribution of assets, tailored to meet specific financial goals, whether they prioritize stability, growth, or a balance of both.

  <img width="635" alt="Ảnh màn hình 2024-08-20 lúc 15 55 55" src="https://github.com/user-attachments/assets/646b1159-2b43-4977-b630-f6008be65d96">

FUTURE WORK

  In the future, we will adjust our investment strategy based on the newest statistical data analyzed by our program. 
  Then we will add more metrics to help us optimize the portfolio. Meanwhile, we will introduce an automated trading module to realize automated trading. 
  From there we will start to monitor our portfolio and continue to optimize, and the optimization direction will start to consider automatic decision making and model speed optimization. 
  Additionally, to better be able to predict future stock prices, we will look at various other machine learning algorithms and weigh the pros and cons of each 
    to see if any other model better suits our needs. A few specific regression algorithms we want to test are the Random Forest Algorithm and the Polynomial Regression algorithm.
  
