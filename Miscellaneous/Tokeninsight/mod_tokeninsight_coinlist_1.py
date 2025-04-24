import requests
import pandas as pd
import json
from pandas import json_normalize

url = "https://api.tokeninsight.com/api/v1/coins/list"

headers = {
    "accept": "application/json",
    "TI_API_KEY": "0513d658e87b461387c332cb73738566"
}

response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Parse JSON response
    data = response.json()
    
    # Extract the coin list (adjust key based on API response structure)
    coins_list = data.get('data', [])
    
    if not coins_list:
        print("No data found in the response.")
    else:
        # Use json_normalize to flatten nested dictionaries
        try:
            df = json_normalize(coins_list)
            
            # Save to CSV
            df.to_csv('coins_list.csv', index=False)
            print("Coin list saved to coins_list.csv")
            
            # Optionally print the first few rows to inspect
            print("Sample data:")
            print(df.head())
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            print("Sample of coins_list for debugging:")
            print(json.dumps(coins_list[:2], indent=2))  # Print first two entries
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
    print(response.text)