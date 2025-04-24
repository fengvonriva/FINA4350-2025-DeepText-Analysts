import requests
import pandas as pd
import os
import glob
from datetime import datetime, date
import json

# Directory containing price CSV files
PRICE_DIR = "P2_Data_Analysis/Pricedata"

# API base URL and headers
BASE_URL = "https://api.tokeninsight.com/api/v1/history/coins/{}?interval=day&length={}"
HEADERS = {
    "accept": "application/json",
    "TI_API_KEY": "0513d658e87b461387c332cb73738566"
}

# Ensure the price directory exists
if not os.path.exists(PRICE_DIR):
    print(f"Directory {PRICE_DIR} does not exist.")
    exit()

# Fetch price data from API for a given coin and number of days
def fetch_price_data(coin_id, days):
    url = BASE_URL.format(coin_id, days)
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            market_chart = data.get('data', {}).get('market_chart', [])
            if market_chart:
                # Convert to DataFrame
                df = pd.DataFrame(market_chart)
                # Convert timestamp to date
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime('%Y-%m-%d')
                # Select and rename columns
                df = df[['date', 'price', 'vol_spot_24h', 'market_cap']]
                # Ensure numeric columns are float
                for col in ['price', 'vol_spot_24h', 'market_cap']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
            else:
                print(f"No market chart data for {coin_id}")
                return None
        else:
            print(f"Failed to fetch data for {coin_id}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching data for {coin_id}: {e}")
        return None

# Update a single CSV file
def update_csv_file(file_path, coin_id):
    try:
        # Read the CSV
        df = pd.read_csv(file_path)
        
        # Ensure date column is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Get the latest date
        latest_date = df['date'].max()
        today = date.today()
        
        # Calculate days difference
        days_difference = (today - latest_date.date()).days
        
        if days_difference <= 1:
            print(f"{coin_id}: Data is up-to-date or only one day behind (latest: {latest_date.date()})")
            return
        
        print(f"{coin_id}: Fetching {days_difference} days of data")
        
        # Fetch new data
        new_data = fetch_price_data(coin_id, days_difference)
        
        if new_data is not None:
            # Filter new data to include only dates after the latest date in CSV
            new_data['date'] = pd.to_datetime(new_data['date'])
            new_data = new_data[new_data['date'] > latest_date]
            
            if new_data.empty:
                print(f"{coin_id}: No new data points after {latest_date.date()}")
                return
            
            # Append new data and remove duplicates
            updated_df = pd.concat([df, new_data]).drop_duplicates(subset='date', keep='last')
            
            # Sort by date
            updated_df['date'] = pd.to_datetime(updated_df['date'])
            updated_df = updated_df.sort_values('date')
            updated_df['date'] = updated_df['date'].dt.strftime('%Y-%m-%d')
            
            # Save to CSV
            updated_df.to_csv(file_path, index=False)
            print(f"Updated {file_path} with {len(new_data)} new data points")
        else:
            print(f"No new data fetched for {coin_id}")
            
    except Exception as e:
        print(f"Error updating {file_path}: {e}")

# Main process
def main():
    # Find all CSV files in the price directory
    csv_files = glob.glob(os.path.join(PRICE_DIR, "*_price.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {PRICE_DIR}")
        return
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        if filename == "terra-luna_price.csv":
            print("Skipping terra-luna_price.csv")
            continue
        
        # Extract coin ID from filename
        coin_id = filename.split('_price')[0]
        print(f"Processing {coin_id}...")
        update_csv_file(file_path, coin_id)

if __name__ == "__main__":
    main()