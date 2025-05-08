import praw
import pandas as pd
import os
import glob
from datetime import datetime, date, timedelta, timezone
import re
import sys
import time
import prawcore

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

# Fetch comments for posts on a specific date
def fetch_comments_for_posts(posts_df, fetch_date):
    try:
        comments = []
        post_urls = ["https://reddit.com" + url for url in posts_df["post_url"].tolist()]
        total_urls = len(post_urls)
        processed_urls = 0
        
        for url in post_urls:
            try:
                time.sleep(2)
                print(f"Fetching comments for post {url} on {fetch_date}", flush=True)
                submission = reddit.submission(url=url)
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list():
                    comment_date = datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).date()
                    if comment_date == fetch_date:
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
            except prawcore.exceptions.ResponseException as e:
                print(f"Error processing post {url}: {e}", flush=True)
                continue
            except Exception as e:
                print(f"Unexpected error processing post {url}: {e}", flush=True)
                continue
        
        if not comments:
            print(f"No comments found for posts on {fetch_date}", flush=True)
            return None
        
        df = pd.DataFrame(comments)
        print(f"Fetched {len(df)} total comments on {fetch_date}", flush=True)
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

# Update comments for a coin
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
        
        # Sort existing comments CSV by date
        if os.path.exists(comment_file):
            comment_df = pd.read_csv(comment_file)
            # Validate date column
            comment_df['date'] = pd.to_datetime(comment_df['date'], format="%Y-%m-%d", errors='coerce')
            invalid_dates = comment_df[comment_df['date'].isna()]
            if not invalid_dates.empty:
                print(f"Found {len(invalid_dates)} invalid dates in {comment_file}: {invalid_dates['date'].tolist()}", flush=True)
                comment_df = comment_df.dropna(subset=['date'])
            comment_df = comment_df.sort_values('date', ascending=True)
            comment_df['date'] = comment_df['date'].dt.strftime('%Y-%m-%d')
            comment_df.to_csv(comment_file, index=False)
            print(f"Sorted and cleaned {comment_file} by date", flush=True)
        
        # Read posts CSV
        post_df = pd.read_csv(post_file)
        post_df['date'] = pd.to_datetime(post_df['date'], format="%Y-%m-%d", errors='coerce')
        post_df = post_df.dropna(subset=['date'])
        
        # Determine date range
        today = date.today()
        yesterday = today - timedelta(days=1)
        
        if os.path.exists(comment_file):
            comment_df = pd.read_csv(comment_file)
            comment_df['date'] = pd.to_datetime(comment_df['date'], format="%Y-%m-%d", errors='coerce')
            comment_df = comment_df.dropna(subset=['date'])
            latest_comment_date = comment_df['date'].max().date()
            start_date = latest_comment_date + timedelta(days=1)
        else:
            start_date = yesterday - timedelta(days=365)  # 1 year back for first run
        
        if start_date > yesterday:
            print(f"{coin_id}: Comment data is up-to-date (latest: {latest_comment_date})", flush=True)
            return total_comments_counter
        
        days_to_fetch = (yesterday - start_date).days + 1
        print(f"{coin_id}: Fetching {days_to_fetch} days of comments", flush=True)
        
        # Fetch comments day by day
        new_comments = []
        current_date = start_date
        while current_date <= yesterday:
            print(f"Processing {coin_id} for {current_date}", flush=True)
            date_str = current_date.strftime('%Y-%m-%d')
            posts_for_date = post_df[post_df['date'] == date_str]
            
            if not posts_for_date.empty:
                comments_df = fetch_comments_for_posts(posts_for_date, current_date)
                if comments_df is not None:
                    filtered_comments = filter_comments_for_coin(comments_df, coin_id)
                    if filtered_comments is not None:
                        new_comments.append(filtered_comments)
            
            current_date += timedelta(days=1)
        
        if new_comments:
            new_data = pd.concat(new_comments, ignore_index=True)
            # Validate new data dates
            new_data['date'] = pd.to_datetime(new_data['date'], format="%Y-%m-%d", errors='coerce')
            invalid_dates = new_data[new_data['date'].isna()]
            if not invalid_dates.empty:
                print(f"Found {len(invalid_dates)} invalid dates in new comments: {invalid_dates['date'].tolist()}", flush=True)
                new_data = new_data.dropna(subset=['date'])
            new_data['date'] = new_data['date'].dt.strftime('%Y-%m-%d')
            
            # Load existing CSV or create new
            if os.path.exists(comment_file):
                comment_df = pd.read_csv(comment_file)
                comment_df['date'] = pd.to_datetime(comment_df['date'], format="%Y-%m-%d", errors='coerce')
                comment_df = comment_df.dropna(subset=['date'])
            else:
                comment_df = pd.DataFrame(columns=['date', 'comment', 'subreddit', 'comment_id', 'upvotes', 'post_url'])
            
            # Append and remove duplicates
            concat_dfs = [df for df in [comment_df, new_data] if not df.empty]
            if concat_dfs:
                updated_df = pd.concat(concat_dfs).drop_duplicates(subset='comment_id', keep='last')
                updated_df['date'] = pd.to_datetime(updated_df['date'], format="%Y-%m-%d", errors='coerce')
                updated_df = updated_df.dropna(subset=['date'])
                updated_df = updated_df.sort_values('date', ascending=True)
                updated_df['date'] = updated_df['date'].dt.strftime('%Y-%m-%d')
                updated_df.to_csv(comment_file, index=False)
                total_comments_counter += len(new_data)
                print(f"Updated {comment_file} with {len(new_data)} new comments", flush=True)
            else:
                print(f"No valid new comments for {coin_id} after cleaning", flush=True)
        else:
            print(f"No new comments for {coin_id}", flush=True)
        
        return total_comments_counter
    except Exception as e:
        print(f"Error updating comments for {coin_id}: {e}", flush=True)
        return total_comments_counter

# Main process
def main():
    # Test authentication
    print("Testing authentication...")
    try:
        print(f"Authenticated as: {reddit.user.me()}", flush=True)
    except Exception as e:
        print(f"Authentication failed: {e}", flush=True)
        exit()
    
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