import yfinance as yf
import pandas as pd
import datetime

def download_btc_price():
    # Calculate time range (consistent with reddit data)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365*2)
    
    print(f"Downloading Bitcoin price data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    # Download BTC-USD data
    btc = yf.download('BTC-USD', 
                      start=start_date,
                      end=end_date,
                      interval='1d')
    # Time zone handling
    btc.index = btc.index.tz_localize(None)
    
    # Save data
    output_path = 'BTC/BTC DATA/btc_price_data.csv'
    btc.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    # Ensure yfinance is installed
    try:
        import yfinance
    except ImportError:
        print("Installing yfinance package...")
        import pip
        pip.main(['install', 'yfinance'])
        
    # Download price data
    btc_price_data = download_btc_price()
