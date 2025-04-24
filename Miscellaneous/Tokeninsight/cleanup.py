import pandas as pd
import os
import glob

# Directory containing price CSV files
PRICE_DIR = "P2_Data_Analysis/Pricedata"

# Ensure the price directory exists
if not os.path.exists(PRICE_DIR):
    print(f"Directory {PRICE_DIR} does not exist.")
    exit()

# Clean up a single CSV file
def clean_csv_file(file_path):
    try:
        # Read the CSV
        df = pd.read_csv(file_path)
        
        # Rename columns (case-insensitive)
        column_map = {
            'Date': 'date',
            'Price': 'price',
            'Volume': 'vol_spot_24h',
            'Market_cap': 'market_cap'
        }
        df.columns = [column_map.get(col, col) for col in df.columns]
        
        # Keep only required columns
        required_columns = ['date', 'price', 'vol_spot_24h', 'market_cap']
        df = df[[col for col in required_columns if col in df.columns]]
        
        # Convert date to yyyy-mm-dd format
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Ensure numeric columns are float
        for col in ['price', 'vol_spot_24h', 'market_cap']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing data
        df = df.dropna()
        
        # Save cleaned CSV
        df.to_csv(file_path, index=False)
        print(f"Cleaned {file_path}")
        return df
    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")
        return None

# Main process
def main():
    # Find all CSV files in the price directory
    csv_files = glob.glob(os.path.join(PRICE_DIR, "*_price.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {PRICE_DIR}")
        return
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        clean_csv_file(file_path)

if __name__ == "__main__":
    main()