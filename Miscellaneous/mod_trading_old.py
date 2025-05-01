# Trading simulator for cryptocurrency based on sentiment analysis
import pandas as pd
import matplotlib.pyplot as plt

# Filepath to the Excel file
filepath = "P2_Data_Analysis/price_x_sentiment/LUNA_price_x_sentiment_20250417211105.xlsx"

try:
    # Read Excel file
    df = pd.read_excel(filepath)
except FileNotFoundError:
    print(f"Error: The file {filepath} was not found. Please check the filepath.")
    exit(1)
except Exception as e:
    print(f"Error reading the Excel file: {e}")
    exit(1)

# Validate required columns
required_columns = ['Date', 'Price', 'avg_afinn_score']
if not all(col in df.columns for col in required_columns):
    print(f"Error: The Excel file must contain the columns: {required_columns}")
    exit(1)

# Convert avg_afinn_score to sentiment (Positive if > 0, Negative if <= 0)
df['Sentiment'] = df['avg_afinn_score'].apply(lambda x: 'Positive' if x > 0 else 'Negative')

# Ensure Date is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Trading simulator
def trading_simulator(dataframe, initial_cash=10000):
    cash = initial_cash
    holdings = 0  # Number of crypto units
    performance_data = []
    
    print("Starting simulation with ${:.2f}".format(initial_cash))
    print("Date\t\tPrice\tSentiment\tAction\t\tCash\tHoldings\tPortfolio Value")
    
    for i in range(len(dataframe)):
        date = dataframe['Date'].iloc[i]
        price = dataframe['Price'].iloc[i]
        sentiment = dataframe['Sentiment'].iloc[i]
        prev_sentiment = dataframe['Sentiment'].iloc[i-1] if i > 0 else None
        
        action = "Hold"
        
        # Trading rules
        if i == 0:
            # Day 1
            if sentiment == 'Positive':
                # Buy as much as possible
                holdings = cash / price
                cash = 0
                action = "Buy"
        else:
            # Subsequent days
            if prev_sentiment == 'Negative' and sentiment == 'Positive':
                # Sentiment switched to positive: Buy
                holdings = cash / price
                cash = 0
                action = "Buy"
            elif prev_sentiment == 'Positive' and sentiment == 'Negative' and holdings > 0:
                # Sentiment switched to negative and holding: Sell
                cash = holdings * price
                holdings = 0
                action = "Sell"
            # Else: Hold (do nothing if already holding or not holding)
        
        # Calculate portfolio value
        portfolio_value = cash + (holdings * price)
        
        # Store performance data
        performance_data.append({
            'Date': date,
            'Price': price,
            'Sentiment': sentiment,
            'Action': action,
            'Cash': cash,
            'Holdings': holdings,
            'Portfolio_Value': portfolio_value
        })
        
        # Output daily status
        print(f"{date.date()}\t${price:.2f}\t{sentiment}\t\t{action}\t\t${cash:.2f}\t{holdings:.4f}\t${portfolio_value:.2f}")
    
    # Final results
    final_value = cash + (holdings * dataframe['Price'].iloc[-1])
    profit = final_value - initial_cash
    print("\nSimulation Complete")
    print(f"Final Cash: ${cash:.2f}")
    print(f"Final Holdings: {holdings:.4f} units")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Profit/Loss: ${profit:.2f} ({(profit/initial_cash)*100:.2f}%)")
    
    # Convert performance data to DataFrame
    performance_df = pd.DataFrame(performance_data)
    
    # Save performance data to CSV
    performance_df.to_csv("P2_Data_Analysis/Trading/trading_performance.csv", index=False)
    print("\nPerformance data saved to 'trading_performance.csv'")
    
    # Plot portfolio value over time
    plt.figure(figsize=(10, 6))
    plt.plot(performance_df['Date'], performance_df['Portfolio_Value'], 'b-', label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('P2_Data_Analysis/Trading/portfolio_performance.png')
    plt.close()
    print("Portfolio performance plot saved to 'portfolio_performance.png'")
    
    return performance_df['Portfolio_Value'].tolist()

# Run the simulator
portfolio_values = trading_simulator(df)