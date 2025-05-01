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
def fetch_posts(subreddit_name, coin_id):
    try:
        print(f"Attempting to fetch posts for {coin_id} from r/{subreddit_name}", flush=True)
        subreddit = reddit.subreddit(subreddit_name)
        subreddit.id  # Verify subreddit exists
        names_tickers = COIN_MAP.get(coin_id, [])
        search_terms = names_tickers
        posts = {"date": [], "title": [], "post_url": [], "subreddit": [], "post_id": []}
        
        # Date range: 2024-04-29 to 2025-04-28
        start_date = datetime(2024, 4, 29, tzinfo=timezone.utc)
        end_date = datetime(2025, 4, 28, 23, 59, 59, tzinfo=timezone.utc)
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        for term in search_terms:
            print(f"Searching r/{subreddit_name} for term '{term}'", flush=True)
            search_results = subreddit.search(term, time_filter="year", limit=1000)
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
                print(f"No posts found for term '{term}' in r/{subreddit_name}", flush=True)
        
        if not posts["title"]:
            print(f"No posts found for {coin_id} in r/{subreddit_name}", flush=True)
            return None
        
        df = pd.DataFrame(posts)
        print(f"Fetched {len(df)} posts for {coin_id} from r/{subreddit_name}", flush=True)
        return df
    except Exception as e:
        print(f"Error fetching posts from r/{subreddit_name} for {coin_id}: {e}", flush=True)
        return None

# Save posts for a coin
def save_posts(coin_id, total_posts_counter):
    try:
        # Read corresponding price CSV to ensure coin exists
        price_file = os.path.join(PRICE_DIR, f"{coin_id}_price.csv")
        if not os.path.exists(price_file):
            print(f"Price file for {coin_id} not found", flush=True)
            return total_posts_counter
        
        post_file = os.path.join(SENTIMENT_DIR, f"{coin_id}_posts.csv")
        posts_dfs = []
        
        # Fetch posts from both subreddits
        for subreddit in [BASE_SUBREDDIT, COIN_SUBREDDITS.get(coin_id, '')]:
            if subreddit:
                posts_df = fetch_posts(subreddit, coin_id)
                if posts_df is not None:
                    posts_dfs.append(posts_df)
        
        if not posts_dfs:
            print(f"No posts found for {coin_id}", flush=True)
            return total_posts_counter
        
        # Combine and deduplicate
        new_data = pd.concat(posts_dfs).drop_duplicates(subset='post_url')
        new_data['date'] = pd.to_datetime(new_data['date']).dt.strftime('%Y-%m-%d')
        
        # Save to CSV
        new_data.to_csv(post_file, index=False)
        total_posts_counter += len(new_data)
        print(f"Saved {post_file} with {len(new_data)} posts", flush=True)
        
        return total_posts_counter
    except Exception as e:
        print(f"Error saving posts for {coin_id}: {e}", flush=True)
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
        
        print(f"Processing {coin_id}...", flush=True)
        total_posts_counter = save_posts(coin_id, total_posts_counter)
    
    print(f"Total posts fetched: {total_posts_counter}", flush=True)

if __name__ == "__main__":
    main()