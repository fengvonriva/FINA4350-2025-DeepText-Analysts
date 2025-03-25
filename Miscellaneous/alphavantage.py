import requests
import pandas as pd

# For future purposes, a standardized data converter is needed that converts the data based on the API function used

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
apikey = "GMWJG3WQVRZR4QL8"

def data_access(data_desired, crypto, fiat):
    #This function carries out the desired API command on Alpha Vantage  

    if data_desired == "current_rate": #getting current exchange rate
        url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={crypto}&to_currency={fiat}&apikey={apikey}'
        r = requests.get(url)
        data = r.json()
        print(url)
        print(data)

    elif data_desired == "historical_daily":
        url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={crypto}&market={fiat}&apikey={apikey}'
        r = requests.get(url)
        data = r.json()

        # Check if the data is valid
        if "Time Series (Digital Currency Daily)" not in data:
            print("Error: No time series data found. Check API key or response:", data)
        else:
            # Extract the time series data
            time_series = data["Time Series (Digital Currency Daily)"]

            # Convert to a list of dictionaries for easier DataFrame creation
            rows = []
            for date, values in time_series.items():
                row = {
                    "date": date,
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "volume": float(values["5. volume"])
                }
                rows.append(row)

            # Create a DataFrame
            df = pd.DataFrame(rows)

            # Sort by date (optional, for chronological order)
            df = df.sort_values("date")

            # Save to CSV
            df.to_csv(f"{crypto}_daily_prices.csv", index=False)
            print(f"Data saved to '{crypto}_daily_prices.csv'")

            # Optional: Print the first few rows to verify
            print("\nFirst 5 rows of the data:")
            print(df.head())
        



if __name__ == "__main__":
    crypto = "LUNA"
    fiat = "USD"
    data_desired = "historical_daily"
    data_access(data_desired, crypto, fiat)