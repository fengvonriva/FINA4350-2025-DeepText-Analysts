import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Set working directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import sentiment analysis methods
from P2_Data_Analysis.Sentiment_Methods.mod_Loughran import analyze_loughran_mcdonald
from P2_Data_Analysis.Sentiment_Methods.mod_AFINN import analyze_afinn

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Configuration
CRYPTO_ID = 'bitcoin'  # TokenInsight ID (e.g., 'BTC' for Bitcoin)
START_DATE = '2024-04-30'  # Selected timeframe start
END_DATE = '2025-04-28'  # Selected timeframe end
SENTIMENT_METHOD = 'AFINN'  # Sentiment analysis method
SENTIMENT_SOURCE = 'comments' # 'comments' or 'news'
TIME_LAG = 1  # Number of days to lag sentiment (default: 1)
INITIAL_CASH = 10000  # Starting capital
SENTIMENT_COLUMN = 'comment'  # Column name for sentiment analysis (Reddit comments)
SAVE_SENTIMENT_DATA = False  # Toggle to save sentiment data file (True/False)

# Filepaths and output folder
SENTIMENT_FILE = f'P2_Data_Analysis/Sentimentdata/{CRYPTO_ID}_{SENTIMENT_SOURCE}.csv'
PRICE_FILE = f'P2_Data_Analysis/Pricedata/{CRYPTO_ID}_price.csv'
TIMESTAMP = datetime.now().strftime('%Y%m%d%H%M%S')
OUTPUT_FOLDER = f'P2_Data_Analysis/Trading_Simulation/{CRYPTO_ID}_{SENTIMENT_METHOD}_trading-simulation_{TIMESTAMP}'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global sentiment dataframe
sentiment_dataframe = None

# Preprocess text (lowercase, remove stopwords, tokenize)
def preprocess_text(text):
    """
    Preprocess text for sentiment analysis.
    Args:
        text (str): Raw text.
    Returns:
        str: Preprocessed text (lowercase, no stopwords, space-separated).
    """
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

# Sentiment analysis function
def analyze_sentiment(sentiment_dataframe, sentiment_column_name, sentiment_method):
    """
    Analyze sentiment for each row in the dataframe and attach scores.
    Args:
        sentiment_dataframe (pd.DataFrame): DataFrame with sentiment data.
        sentiment_column_name (str): Column containing text to analyze.
        sentiment_method (str): Sentiment analysis method ('AFINN', 'LoughranMcDonald').
    Returns:
        pd.DataFrame: DataFrame with added 'method_score' and 'standardized_score' columns.
    """
    df = sentiment_dataframe.copy()
    
    # Map sentiment method to function
    method_map = {
        'AFINN': analyze_afinn,
        'LoughranMcDonald': analyze_loughran_mcdonald
    }
    
    if sentiment_method not in method_map:
        raise ValueError(f"Unsupported sentiment method: {sentiment_method}")
    
    # Apply sentiment analysis
    def get_scores(text):
        score = method_map[sentiment_method](text)
        standardized_score = 1 if score > 0 else -1 if score < 0 else 0
        return pd.Series([score, standardized_score], index=['method_score', 'standardized_score'])
    
    df[['method_score', 'standardized_score']] = df[sentiment_column_name].apply(get_scores)
    
    return df

# Load and process sentiment data
def load_sentiment_data():
    global sentiment_dataframe
    try:
        sentiment_dataframe = pd.read_csv(SENTIMENT_FILE)
    except FileNotFoundError:
        print(f"Error: Sentiment file {SENTIMENT_FILE} not found.")
        exit(1)
    
    # Validate columns
    required_cols = ['date', 'comment', 'subreddit', 'comment_id', 'upvotes', 'post_url']
    if not all(col in sentiment_dataframe.columns for col in required_cols):
        print(f"Error: Sentiment file must contain columns: {required_cols}")
        exit(1)
    
    # Preprocess comments
    sentiment_dataframe['preprocessed_comment'] = sentiment_dataframe['comment'].apply(preprocess_text)
    sentiment_dataframe['date'] = pd.to_datetime(sentiment_dataframe['date'])
    sentiment_dataframe = sentiment_dataframe[
        (sentiment_dataframe['date'] >= pd.to_datetime(START_DATE)) & 
        (sentiment_dataframe['date'] <= pd.to_datetime(END_DATE))
    ]
    
    # Analyze sentiment
    sentiment_dataframe = analyze_sentiment(
        sentiment_dataframe, 
        sentiment_column_name='preprocessed_comment', 
        sentiment_method=SENTIMENT_METHOD
    )
    
    # Calculate weighted score (standardized_score * upvotes)
    sentiment_dataframe['weighted_score'] = sentiment_dataframe['standardized_score'] * sentiment_dataframe['upvotes']
    
    # Save scored sentiment data (if enabled)
    if SAVE_SENTIMENT_DATA:
        sentiment_output_file = f'{OUTPUT_FOLDER}/{CRYPTO_ID}_{SENTIMENT_METHOD}_sentiment-data_{TIMESTAMP}.csv'
        sentiment_dataframe.to_csv(sentiment_output_file, index=False)
        print(f"Scored sentiment data saved to {sentiment_output_file}")
    
    # Aggregate daily sentiment
    daily_sentiment = sentiment_dataframe.groupby(sentiment_dataframe['date'].dt.date).agg({
        'weighted_score': 'sum'
    }).reset_index()
    daily_sentiment['sentiment'] = daily_sentiment['weighted_score'].apply(
        lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral'
    )
    daily_sentiment['sentiment_score'] = daily_sentiment['weighted_score']
    
    return daily_sentiment

# Load price data
def load_price_data():
    try:
        df = pd.read_csv(PRICE_FILE)
    except FileNotFoundError:
        print(f"Error: Price file {PRICE_FILE} not found.")
        exit(1)
    
    # Validate columns
    required_cols = ['date', 'price', 'vol_spot_24h', 'market_cap']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Price file must contain columns: {required_cols}")
        exit(1)
    
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= pd.to_datetime(START_DATE)) & (df['date'] <= pd.to_datetime(END_DATE))]
    return df[['date', 'price']]

# Combine price and sentiment data
def combine_data(price_df, sentiment_df):
    price_df['date'] = price_df['date'].dt.date
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date  # Convert to date object
    combined_df = price_df.merge(sentiment_df[['date', 'sentiment', 'sentiment_score']], 
                                on='date', how='left')
    combined_df['sentiment'] = combined_df['sentiment'].fillna('Neutral')
    combined_df['sentiment_score'] = combined_df['sentiment_score'].fillna(0)
    
    # Save price-sentiment data
    output_file = f'{OUTPUT_FOLDER}/{CRYPTO_ID}_{SENTIMENT_METHOD}_price-x-sentiment_{TIMESTAMP}.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"Price-sentiment data saved to {output_file}")
    
    return combined_df

# Trading simulator
def trading_simulator(data, method, time_lag):
    cash = INITIAL_CASH
    holdings = 0  # Units of crypto (long position)
    short_holdings = 0  # Units of crypto (short position, for Method 2)
    short_entry_price = 0  # Price at which short position was opened
    performance = []
    
    initial_price = data['price'].iloc[0]
    
    for i in range(time_lag, len(data)):
        date = data['date'].iloc[i]
        price = data['price'].iloc[i]
        sentiment = data['sentiment'].iloc[i - time_lag]
        sentiment_score = data['sentiment_score'].iloc[i - time_lag]
        
        action = 'Hold'
        
        if method == 1:
            # Method 1: Long-only
            if sentiment == 'Positive' and holdings == 0:
                holdings = cash / price
                cash = 0
                action = 'Buy'
            elif sentiment == 'Negative' and holdings > 0:
                cash = holdings * price
                holdings = 0
                action = 'Sell'
        else:
            # Method 2: Long and short
            if sentiment == 'Positive' and holdings == 0 and short_holdings == 0:
                holdings = cash / price
                cash = 0
                action = 'Buy Long'
            elif sentiment == 'Positive' and short_holdings > 0:
                # Close short position (buy back)
                gain_loss = short_holdings * (short_entry_price - price)
                cash = INITIAL_CASH + gain_loss  # Adjust cash based on gain/loss
                if cash < 0:
                    cash = 0  # Prevent negative cash (simulating margin call)
                short_holdings = 0
                short_entry_price = 0
                action = 'Close Short'
            elif sentiment == 'Negative' and holdings > 0:
                cash = holdings * price
                holdings = 0
                action = 'Sell Long'
            elif sentiment == 'Negative' and holdings == 0 and short_holdings == 0:
                # Short the entire portfolio value
                short_holdings = INITIAL_CASH / price
                short_entry_price = price
                cash = 0  # All cash is used for the short position
                action = 'Sell Short'
        
        # Calculate portfolio value
        if short_holdings > 0:
            # Portfolio value = proceeds from short sale - current value of shorted assets
            portfolio_value = short_holdings * (short_entry_price - price)
            # Simulate margin call: if portfolio value goes too negative, close position
            if portfolio_value < -INITIAL_CASH:
                gain_loss = short_holdings * (short_entry_price - price)
                cash = INITIAL_CASH + gain_loss
                if cash < 0:
                    cash = 0
                short_holdings = 0
                short_entry_price = 0
                action = 'Close Short (Margin Call)'
                portfolio_value = cash
        else:
            # Portfolio value based on long position or cash
            portfolio_value = cash + (holdings * price)
        
        crypto_return = (price - initial_price) / initial_price * 100
        
        performance.append({
            'Date': date,
            'Price': price,
            'Sentiment': sentiment,
            'Sentiment_Score': sentiment_score,
            'Action': action,
            'Cash': cash,
            'Holdings': holdings,
            'Short_Holdings': short_holdings,
            'Portfolio_Value': portfolio_value,
            'Crypto_Return': crypto_return
        })
    
    performance_df = pd.DataFrame(performance)
    portfolio_return = (performance_df['Portfolio_Value'].iloc[-1] - INITIAL_CASH) / INITIAL_CASH * 100
    performance_df['Portfolio_Return'] = performance_df['Portfolio_Value'].apply(
        lambda x: (x - INITIAL_CASH) / INITIAL_CASH * 100 if x >= 0 else -100
    )
    
    return performance_df, portfolio_return

# Plot and save results
def plot_results(method1_df, method1_return, method2_df, method2_return):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Primary y-axis: Cumulative Return (%)
    ax1.plot(method1_df['Date'], method1_df['Portfolio_Return'], label=f'Method 1 Return ({method1_return:.2f}%)', color='blue')
    # ax1.plot(method2_df['Date'], method2_df['Portfolio_Return'], label=f'Method 2 Return ({method2_return:.2f}%)', color='green')
    ax1.plot(method1_df['Date'], method1_df['Crypto_Return'], label='Crypto Return', color='orange', linestyle='--')

    # Annotate sentiment scores
    for i, row in method1_df.iterrows():
        if abs(row['Sentiment_Score']) > 0:  # Only annotate non-zero scores
            ax1.text(row['Date'], row['Portfolio_Return'], f'{int(row["Sentiment_Score"])}', fontsize=8)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return (%)', color='black')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='x', rotation=45)

    # Secondary y-axis: Portfolio Value ($)
    ax2 = ax1.twinx()
    ax2.plot(method1_df['Date'], method1_df['Portfolio_Value'], label='Method 1 Value', color='blue', linestyle='-.', alpha=0.5)
    # ax2.plot(method2_df['Date'], method2_df['Portfolio_Value'], label='Method 2 Value', color='green', linestyle='-.', alpha=0.5)
    # Calculate buy-and-hold portfolio value
    crypto_value = (method1_df['Price'] / method1_df['Price'].iloc[0]) * INITIAL_CASH
    ax2.plot(method1_df['Date'], crypto_value, label='Crypto Value', color='orange', linestyle=':', alpha=0.5)

    ax2.set_ylabel('Portfolio Value ($)', color='black')
    ax2.legend(loc='upper right')

    plt.title(f'{CRYPTO_ID} Trading Performance ({SENTIMENT_METHOD})')
    plt.tight_layout()
    
    output_file = f'{OUTPUT_FOLDER}/{CRYPTO_ID}_{SENTIMENT_METHOD}_trading-simulation_{TIMESTAMP}.png'
    plt.savefig(output_file)
    plt.close()
    print(f"Trading performance plot saved to {output_file}")

def calculate_sharpe_ratio(returns, periods_per_year=252, risk_free_rate=0):
    """
    Calculate the Sharpe ratio for a series of daily returns.
    Args:
        returns (pd.Series): Daily returns.
        periods_per_year (int): Number of trading periods per year (default: 252).
        risk_free_rate (float): Annual risk-free rate (default: 0).
    Returns:
        float: Sharpe ratio (annualized).
    """
    if len(returns) < 2:
        return 0
    mean_daily_return = returns.mean()
    annualized_return = mean_daily_return * periods_per_year
    daily_volatility = returns.std()
    annualized_volatility = daily_volatility * (periods_per_year ** 0.5)
    if annualized_volatility == 0 or pd.isna(annualized_volatility):
        return 0  # Avoid division by zero
    sharpe = (annualized_return - risk_free_rate) / annualized_volatility
    return sharpe

def calculate_annualized_return(total_return, days, periods_per_year=252):
    """
    Calculate the annualized return.
    Args:
        total_return (float): Total return (%).
        days (int): Number of days in the period.
        periods_per_year (int): Number of trading periods per year (default: 252).
    Returns:
        float: Annualized return (%).
    """
    if days <= 0:
        return 0
    years = days / periods_per_year
    if years <= 0:
        return 0
    annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
    return annualized_return

def calculate_max_drawdown(portfolio_values):
    """
    Calculate the maximum drawdown.
    Args:
        portfolio_values (pd.Series): Portfolio values over time.
    Returns:
        float: Maximum drawdown (%).
    """
    if len(portfolio_values) < 2:
        return 0
    rolling_max = portfolio_values.cummax()
    drawdowns = (portfolio_values - rolling_max) / rolling_max * 100
    max_drawdown = drawdowns.min()  # Most negative drawdown
    return -max_drawdown  # Return as positive percentage

def calculate_volatility(returns, periods_per_year=252):
    """
    Calculate annualized volatility.
    Args:
        returns (pd.Series): Daily returns.
        periods_per_year (int): Number of trading periods per year (default: 252).
    Returns:
        float: Annualized volatility (%).
    """
    if len(returns) < 2:
        return 0
    daily_volatility = returns.std()
    annualized_volatility = daily_volatility * (periods_per_year ** 0.5) * 100
    return annualized_volatility

def main():
    # Load data
    sentiment_df = load_sentiment_data()
    price_df = load_price_data()
    combined_df = combine_data(price_df, sentiment_df)
    
    # Run trading simulations
    method1_df, method1_return = trading_simulator(combined_df, method=1, time_lag=TIME_LAG)
    method2_df, method2_return = trading_simulator(combined_df, method=2, time_lag=TIME_LAG)
    
    # Calculate buy-and-hold portfolio (buy at start, hold until end)
    buy_hold_value = (combined_df['price'] / combined_df['price'].iloc[0]) * INITIAL_CASH
    buy_hold_df = combined_df.copy()
    buy_hold_df['Portfolio_Value'] = buy_hold_value
    buy_hold_return = (buy_hold_df['Portfolio_Value'].iloc[-1] - INITIAL_CASH) / INITIAL_CASH * 100
    buy_hold_df['Portfolio_Return'] = buy_hold_df['Portfolio_Value'].apply(
        lambda x: (x - INITIAL_CASH) / INITIAL_CASH * 100
    )
    
    # Save performance data
    method1_df.to_csv(f'{OUTPUT_FOLDER}/{CRYPTO_ID}_{SENTIMENT_METHOD}_trading-simulation_method1_{TIMESTAMP}.csv', index=False)
    method2_df.to_csv(f'{OUTPUT_FOLDER}/{CRYPTO_ID}_{SENTIMENT_METHOD}_trading-simulation_method2_{TIMESTAMP}.csv', index=False)
    print(f"Trading performance data saved to {OUTPUT_FOLDER}")
    
    # Plot results
    plot_results(method1_df, method1_return, method2_df, method2_return)
    
    # Calculate performance metrics
    days = (combined_df['date'].iloc[-1] - combined_df['date'].iloc[0]).days
    if days <= 0:
        days = 1  # Avoid division by zero
    
    # Daily returns for each portfolio
    method1_daily_returns = method1_df['Portfolio_Return'].pct_change().dropna()
    method2_daily_returns = method2_df['Portfolio_Return'].pct_change().dropna()
    buy_hold_daily_returns = buy_hold_df['Portfolio_Return'].pct_change().dropna()
    
    # Sharpe Ratio
    method1_sharpe = calculate_sharpe_ratio(method1_daily_returns)
    method2_sharpe = calculate_sharpe_ratio(method2_daily_returns)
    buy_hold_sharpe = calculate_sharpe_ratio(buy_hold_daily_returns)
    
    # Annualized Return
    method1_annual_return = calculate_annualized_return(method1_return, days)
    method2_annual_return = calculate_annualized_return(method2_return, days)
    buy_hold_annual_return = calculate_annualized_return(buy_hold_return, days)
    
    # Maximum Drawdown
    method1_max_drawdown = calculate_max_drawdown(method1_df['Portfolio_Value'])
    method2_max_drawdown = calculate_max_drawdown(method2_df['Portfolio_Value'])
    buy_hold_max_drawdown = calculate_max_drawdown(buy_hold_df['Portfolio_Value'])
    
    # Volatility
    method1_volatility = calculate_volatility(method1_daily_returns)
    method2_volatility = calculate_volatility(method2_daily_returns)
    buy_hold_volatility = calculate_volatility(buy_hold_daily_returns)

    # Debug daily returns
    print(f"Method 1 Daily Returns Length: {len(method1_daily_returns)}, Std: {method1_daily_returns.std()}")
    print(f"Method 2 Daily Returns Length: {len(method2_daily_returns)}, Std: {method2_daily_returns.std()}")
    print(f"Buy-and-Hold Daily Returns Length: {len(buy_hold_daily_returns)}, Std: {buy_hold_daily_returns.std()}")
    
    # Print summary with metrics
    print("\nSimulation Complete")
    print("\nPortfolio Performance Metrics:")
    print("-----------------------------")
    print(f"{'Metric':<25} {'Method 1':<15} {'Method 2':<15} {'Buy-and-Hold':<15}")
    print(f"{'Final Return (%)':<25} {method1_return:<15.2f} {method2_return:<15.2f} {buy_hold_return:<15.2f}")
    print(f"{'Annualized Return (%)':<25} {method1_annual_return:<15.2f} {method2_annual_return:<15.2f} {buy_hold_annual_return:<15.2f}")
    print(f"{'Sharpe Ratio':<25} {method1_sharpe:<15.2f} {method2_sharpe:<15.2f} {buy_hold_sharpe:<15.2f}")
    print(f"{'Maximum Drawdown (%)':<25} {method1_max_drawdown:<15.2f} {method2_max_drawdown:<15.2f} {buy_hold_max_drawdown:<15.2f}")
    print(f"{'Annualized Volatility (%)':<25} {method1_volatility:<15.2f} {method2_volatility:<15.2f} {buy_hold_volatility:<15.2f}")

if __name__ == "__main__":
    main()