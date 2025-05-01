import praw
import pandas as pd
import os
import glob
from datetime import datetime, date, timedelta, timezone
import re
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

# Ensure directories exist
os.makedirs(SENTIMENT_DIR, exist_ok=True)
if not os.path.exists(PRICE_DIR):
    print(f"Price directory {PRICE_DIR} does not exist.", flush=True)
    exit()

# Fetch comments for posts
def fetch_comments_for_posts(posts_df):
    try:
        # Date range: 2024-04-29 to 2025-04-28
        start_date = datetime(2024, 4, 29, tzinfo=timezone.utc)
        end_date = datetime(2025, 4, 28, 23, 59, 59, tzinfo=timezone.utc)
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        comments = []
        post_urls = ["https://reddit.com" + url for url in posts_df["post_url"].tolist()]
        total_urls = len(post_urls)
        processed_urls = 0
        
        for url in post_urls:
            try:
                time.sleep(1)
                print(f"Fetching comments for post {url}", flush=True)
                submission = reddit.submission(url=url)
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list():
                    if start_timestamp <= comment.created_utc <= end_timestamp:
                        cleaned_comment = comment.body.replace('\n', ' ')
                        comments.append({
                            'date': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).strftime('%Y-%m-%d'),
                            'comment': cleaned_comment,
                            'subreddit': submission.subreddit.display_name,
                            'comment_id': comment.id,
                            'upvotes': comment.score,
                            'post_url': url
                        })
                
                processed_urls += 1
                if processed_urls % 10 == 0:
                    print(f"Processed {processed_urls}/{total_urls} posts", flush=True)
            except Exception as e:
                print(f"Error processing post {url}: {e}", flush=True)
                continue
        
        if not comments:
            print(f"No comments found for posts", flush=True)
            return None
        
        df = pd.DataFrame(comments)
        print(f"Fetched {len(df)} total comments", flush=True)
        return df
    except Exception as e:
        print(f"Error fetching comments: {e}", flush=True)
        return None

# Filter comments for a specific coin
def filter_comments_for_coin(df, coin_id):
    if df is None or df.empty:
        return None
    
    names_tickers = COIN_MAP.get(coin_id, [])
    pattern = rf"\b({'|'.join(re.escape(nt) for nt in names_tickers)})\b"
    
    mask = df['comment'].str.contains(pattern, case=False, na=False)
    filtered_df = df[mask].copy()
    
    if filtered_df.empty:
        return None
    
    return filtered_df[['date', 'comment', 'subreddit', 'comment_id', 'upvotes', 'post_url']]

# Save comments for a coin
def save_comments(coin_id, total_comments_counter):
    try:
        # Read corresponding price and post CSV
        price_file = os.path.join(PRICE_DIR, f"{coin_id}_price.csv")
        post_file = os.path.join(SENTIMENT_DIR, f"{coin_id}_posts.csv")
        if not os.path.exists(price_file):
            print(f"Price file for {coin_id} not found", flush=True)
            return total_comments_counter
        if not os.path.exists(post_file):
            print(f"Post file for {coin_id} not found", flush=True)
            return total_comments_counter
        
        comment_file = os.path.join(SENTIMENT_DIR, f"{coin_id}_comments.csv")
        
        # Read posts CSV
        post_df = pd.read_csv(post_file)
        post_df['date'] = pd.to_datetime(post_df['date'])
        
        # Fetch comments
        comments_df = fetch_comments_for_posts(post_df)
        if comments_df is None:
            print(f"No comments found for {coin_id}", flush=True)
            return total_comments_counter
        
        # Filter comments
        filtered_comments = filter_comments_for_coin(comments_df, coin_id)
        if filtered_comments is None:
            print(f"No relevant comments found for {coin_id}", flush=True)
            return total_comments_counter
        
        # Save to CSV
        filtered_comments['date'] = pd.to_datetime(filtered_comments['date']).dt.strftime('%Y-%m-%d')
        filtered_comments.to_csv(comment_file, index=False)
        total_comments_counter += len(filtered_comments)
        print(f"Saved {comment_file} with {len(filtered_comments)} comments", flush=True)
        
        return total_comments_counter
    except Exception as e:
        print(f"Error saving comments for {coin_id}: {e}", flush=True)
        return total_comments_counter

# Main process
def main():
    # Find all price CSV files
    csv_files = glob.glob(os.path.join(PRICE_DIR, "*_price.csv"))
    
    if not csv_files:
        print(f"No price CSV files found in {PRICE_DIR}", flush=True)
        return
    
    total_comments_counter = 0
    
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
        total_comments_counter = save_comments(coin_id, total_comments_counter)
    
    print(f"Total comments fetched: {total_comments_counter}", flush=True)

if __name__ == "__main__":
    main()

# fetched 102,666 comments