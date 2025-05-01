import praw
import pandas as pd
import os
import glob
from datetime import datetime, date, timedelta, timezone
import sys
import time

# Verify Python version
print(f"Python version: {sys.version}", flush=True)
if sys.version_info < (3, 2):
    print("Error: Python 3.2 or higher is required", flush=True)
    exit()

# Reddit API credentials
reddit = praw.Reddit(
    client_id='ZExuVDrnuon1q8SWA__2fw',
    client_secret='TENjvYzdpCZV2tA8gwA_8bEkyNfghg',
    user_agent='ImportanceAsleep6865',
)

# Directories
PRICE_DIR = "P2_Data_Analysis/Pricedata"
SENTIMENT_DIR = "P2_Data_Analysis/Sentimentdata"

# Cryptocurrencies and their names/tickers
COIN_MAP = {
    "binance-coin": ["binance coin", "binance", "BNB"],
    "bitcoin": ["bitcoin", "BTC"],
    "ethereum": ["ethereum", "ETH"],
    "solana": ["solana", "SOL"],
    "cardano": ["cardano", "ADA"],
    "dogecoin": ["dogecoin", "doge", "DOGE"],
    "ripple": ["ripple", "XRP"]
}

# Subreddits to monitor
BASE_SUBREDDIT = "CryptoCurrency"
COIN_SUBREDDITS = {
    "binance-coin": "Binance",
    "bitcoin": "Bitcoin",
    "ethereum": "ethereum",
    "solana": "Solana",
    "cardano": "cardano",
    "dogecoin": "dogecoin",
    "ripple": "Ripple"
}

# Ensure directories exist
os.makedirs(SENTIMENT_DIR, exist_ok=True)
if not os.path.exists(PRICE_DIR):
    print(f"Price directory {PRICE_DIR} does not exist.", flush=True)
    exit()

# Fetch posts for a subreddit and coin
def fetch_posts(subreddit_name, coin_id, fetch_date):
    try:
        print(f"Attempting to fetch posts for {coin_id} from r/{subreddit_name} on {fetch_date}", flush=True)
        subreddit = reddit.subreddit(subreddit_name)
        subreddit.id  # Verify subreddit exists
        names_tickers = COIN_MAP.get(coin_id, [])
        search_terms = names_tickers
        posts = {"date": [], "title": [], "post_url": [], "subreddit": [], "post_id": []}
        
        start_timestamp = int(datetime.combine(fetch_date, datetime.min.time()).timestamp())
        end_timestamp = start_timestamp + 86399
        
        for term in search_terms:
            print(f"Searching r/{subreddit_name} for term '{term}' on {fetch_date}", flush=True)
            search_results = subreddit.search(term, limit=200)
            post_count = 0
            for post in search_results:
                if start_timestamp <= post.created_utc <= end_timestamp:
                    if post.permalink not in posts["post_url"]:
                        posts["date"].append(
                            datetime.fromtimestamp(post.created_utc, tz=timezone.utc).strftime('%Y-%m-%d')
                        )
                        posts["title"].append(post.title)
                        posts["post_url"].append(post.permalink)
                        posts["subreddit"].append(subreddit_name)
                        posts["post_id"].append(post.id)
                        post_count += 1
            if post_count == 0:
                print(f"No posts found for term '{term}' in r/{subreddit_name} on {fetch_date}", flush=True)
        
        if not posts["title"]:
            print(f"No posts found for {coin_id} in r/{subreddit_name} on {fetch_date}", flush=True)
            return None
        
        df = pd.DataFrame(posts)
        print(f"Fetched {len(df)} posts for {coin_id} from r/{subreddit_name} on {fetch_date}", flush=True)
        return df
    except Exception as e:
        print(f"Error fetching posts from r/{subreddit_name} for {coin_id}: {e}", flush=True)
        return None

# Update a single post CSV
def update_post_csv(file_path, coin_id, total_posts_counter):
    try:
        # Read corresponding price CSV to ensure coin exists
        price_file = os.path.join(PRICE_DIR, f"{coin_id}_price.csv")
        if not os.path.exists(price_file):
            print(f"Price file for {coin_id} not found", flush=True)
            return total_posts_counter
        
        # Determine date range
        today = date.today()
        yesterday = today - timedelta(days=1)
        
        if os.path.exists(file_path):
            post_df = pd.read_csv(file_path)
            post_df['date'] = pd.to_datetime(post_df['date'])
            latest_post_date = post_df['date'].max().date()
            start_date = latest_post_date + timedelta(days=1)
        else:
            start_date = yesterday - timedelta(days=365)  # 1 year back for first run
        
        if start_date > yesterday:
            print(f"{coin_id}: Post data is up-to-date (latest: {latest_post_date})", flush=True)
            return total_posts_counter
        
        days_to_fetch = (yesterday - start_date).days + 1
        print(f"{coin_id}: Fetching {days_to_fetch} days of posts", flush=True)
        
        # Fetch posts day by day
        new_posts = []
        current_date = start_date
        while current_date <= yesterday:
            print(f"Processing {coin_id} for {current_date}", flush=True)
            posts_dfs = []
            for subreddit in [BASE_SUBREDDIT, COIN_SUBREDDITS.get(coin_id, '')]:
                if subreddit:
                    posts_df = fetch_posts(subreddit, coin_id, current_date)
                    if posts_df is not None:
                        posts_dfs.append(posts_df)
            
            if posts_dfs:
                new_posts.append(pd.concat(posts_dfs).drop_duplicates(subset='post_url'))
            
            current_date += timedelta(days=1)
        
        if new_posts:
            new_data = pd.concat(new_posts, ignore_index=True)
            new_data['date'] = pd.to_datetime(new_data['date']).dt.strftime('%Y-%m-%d')
            
            # Load existing CSV or create new
            if os.path.exists(file_path):
                post_df = pd.read_csv(file_path)
                post_df['date'] = pd.to_datetime(post_df['date'])
            else:
                post_df = pd.DataFrame(columns=['date', 'title', 'post_url', 'subreddit', 'post_id'])
            
            # Append and remove duplicates
            updated_df = pd.concat([post_df, new_data]).drop_duplicates(subset='post_id', keep='last')
            updated_df['date'] = pd.to_datetime(updated_df['date'])
            
            # Sort by date
            updated_df = updated_df.sort_values('date', ascending=True)
            updated_df['date'] = updated_df['date'].dt.strftime('%Y-%m-%d')
            
            # Update counter
            total_posts_counter += len(new_data)
            
            # Save to CSV
            updated_df.to_csv(file_path, index=False)
            print(f"Updated {file_path} with {len(new_data)} new posts", flush=True)
        else:
            print(f"No new posts for {coin_id}", flush=True)
            
        return total_posts_counter
    except Exception as e:
        print(f"Error updating {file_path}: {e}", flush=True)
        return total_posts_counter

# Main process
def main():
    # Find all price CSV files
    csv_files = glob.glob(os.path.join(PRICE_DIR, "*_price.csv"))
    
    if not csv_files:
        print(f"No price CSV files found in {PRICE_DIR}", flush=True)
        return
    
    total_posts_counter = 0
    
    for price_file in csv_files:
        filename = os.path.basename(price_file)
        if filename == "terra-luna_price.csv":
            print("Skipping terra-luna_price.csv", flush=True)
            continue
        
        # Extract coin ID
        coin_id = filename.split('_price')[0]
        if coin_id not in COIN_MAP:
            print(f"Skipping {coin_id}: Not in supported coin list", flush=True)
            continue
        
        post_file = os.path.join(SENTIMENT_DIR, f"{coin_id}_posts.csv")
        print(f"Processing {coin_id}...", flush=True)
        total_posts_counter = update_post_csv(post_file, coin_id, total_posts_counter)
    
    print(f"Total posts fetched: {total_posts_counter}", flush=True)

if __name__ == "__main__":
    main()