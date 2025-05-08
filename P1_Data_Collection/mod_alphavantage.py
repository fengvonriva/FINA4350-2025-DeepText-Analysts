import requests
import pandas as pd
import os
from datetime import datetime, date, timedelta
import sys
import time
import json
import re

# Verify Python version
print(f"Python version: {sys.version}", flush=True)
if sys.version_info < (3, 2):
    print("Error: Python 3.2 or higher is required", flush=True)
    exit()

# Alpha Vantage API key
API_KEY = "683SUL8BPL2RK67T"

# Directory
SENTIMENT_DIR = "P2_Data_Analysis/Sentimentdata"

# Cryptocurrencies and their tickers
COIN_MAP = {
    "binance-coin": "CRYPTO:BNB",
    "bitcoin": "CRYPTO:BTC",
    "ethereum": "CRYPTO:ETH",
    "solana": "CRYPTO:SOL",
    "cardano": "CRYPTO:ADA",
    "dogecoin": "CRYPTO:DOGE",
    "ripple": "CRYPTO:XRP"
}

# Ensure directory exists
os.makedirs(SENTIMENT_DIR, exist_ok=True)

# Validate and extract date from time_published for Reddit-compatible
def extract_date(time_published):
    try:
        # Expect format like YYYYMMDDTHHMMSS, YYYY-MM-DDTHH:MM:SS, or partial
        if not isinstance(time_published, str):
            print(f"Invalid time_published type: {time_published}", flush=True)
            return None
        # Match YYYYMMDD or YYYY-MM-DD
        match = re.match(r"^(\d{4})(\d{2})(\d{2})T|\d{4}-\d{2}-\d{2}", time_published)
        if not match:
            print(f"Invalid time_published format: {time_published}", flush=True)
            return None
        # Extract or format YYYY-MM-DD
        date_str = time_published[:10]
        if len(date_str) == 8:  # YYYYMMDD
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        elif len(date_str) == 10 and date_str[4] != '-' and date_str[7] != '-':
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        # Validate date
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        # Filter out future dates
        if parsed_date > date.today():
            print(f"Skipping future date: {date_str}", flush=True)
            return None
        return date_str
    except Exception as e:
        print(f"Error parsing date {time_published}: {e}", flush=True)
        return None

# Fetch news sentiment data
def fetch_news(ticker, time_from):
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={time_from}&limit=1000&sort=LATEST&apikey={API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if "feed" not in data or not data["feed"]:
            print(f"No news found for {ticker} from {time_from}", flush=True)
            return None, None
        
        # Log sample time_published values
        sample_times = [article["time_published"] for article in data["feed"][:3]]
        print(f"Sample time_published values for {ticker}: {sample_times}", flush=True)
        
        # Reddit-compatible DataFrame
        news_items = []
        for article in data["feed"]:
            date_str = extract_date(article["time_published"])
            if date_str is None:
                continue
            
            # Find relevance score for the ticker of interest
            relevance_score = 0.0
            for ts in article["ticker_sentiment"]:
                if ts["ticker"] == ticker:
                    relevance_score = float(ts["relevance_score"])
                    break
            
            news_items.append({
                "date": date_str,
                "comment": article["summary"],
                "subreddit": article["source"],
                "comment_id": article["url"],
                "upvotes": relevance_score,
                "post_url": article["url"],
                "title": article["title"],
                "authors": ",".join(article["authors"]),
                "banner_image": article.get("banner_image", ""),
                "source_domain": article["source_domain"],
                "topics": ",".join([t["topic"] for t in article["topics"]]),
                "overall_sentiment_score": article["overall_sentiment_score"],
                "overall_sentiment_label": article["overall_sentiment_label"],
                "ticker_sentiment": json.dumps(article["ticker_sentiment"])
            })
        
        reddit_df = pd.DataFrame(news_items) if news_items else pd.DataFrame(columns=[
            'date', 'comment', 'subreddit', 'comment_id', 'upvotes', 'post_url',
            'title', 'authors', 'banner_image', 'source_domain', 'topics',
            'overall_sentiment_score', 'overall_sentiment_label', 'ticker_sentiment'
        ])
        print(f"Reddit-compatible DataFrame size: {len(reddit_df)} rows", flush=True)
        
        # Raw DataFrame (preserve all articles)
        raw_items = []
        for article in data["feed"]:
            article_copy = article.copy()
            # JSON-encode nested fields
            article_copy["topics"] = json.dumps(article_copy["topics"])
            article_copy["ticker_sentiment"] = json.dumps(article_copy["ticker_sentiment"])
            article_copy["authors"] = json.dumps(article_copy["authors"])
            raw_items.append(article_copy)
        
        raw_df = pd.DataFrame(raw_items) if raw_items else pd.DataFrame(columns=[
            'title', 'url', 'time_published', 'authors', 'summary', 'banner_image',
            'source', 'category_within_source', 'source_domain', 'topics',
            'overall_sentiment_score', 'overall_sentiment_label', 'ticker_sentiment'
        ])
        print(f"Raw DataFrame size: {len(raw_df)} rows", flush=True)
        
        if raw_df.empty:
            print(f"Warning: Raw DataFrame is empty for {ticker} from {time_from}", flush=True)
        
        return reddit_df, raw_df
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}", flush=True)
        return None, None

# Update news CSVs for a coin
def update_news_csv(coin_id, total_news_counter):
    try:
        news_file = os.path.join(SENTIMENT_DIR, f"{coin_id}_news.csv")
        raw_news_file = os.path.join(SENTIMENT_DIR, f"{coin_id}_news_raw.csv")
        ticker = COIN_MAP.get(coin_id)
        
        # Determine start date
        start_date = date(2024, 4, 30)  # Default start
        if os.path.exists(news_file):
            news_df = pd.read_csv(news_file)
            news_df['date'] = pd.to_datetime(news_df['date'], format="%Y-%m-%d", errors='coerce')
            news_df = news_df.dropna(subset=['date'])
            if not news_df.empty:
                latest_news_date = news_df['date'].max().date()
                start_date = latest_news_date + timedelta(days=1)
        
        # Skip API call if start_date is in the future
        if start_date > date.today():
            print(f"Skipping API call for {coin_id}: Data up-to-date until {latest_news_date}", flush=True)
            return total_news_counter
        
        time_from = start_date.strftime("%Y%m%dT0000")
        reddit_df, raw_df = fetch_news(ticker, time_from)
        
        if reddit_df is None and raw_df is None:
            print(f"No new news for {coin_id} from {time_from}", flush=True)
            return total_news_counter
        
        # Process Reddit-compatible
        if reddit_df is not None and not reddit_df.empty:
            reddit_df['date'] = pd.to_datetime(reddit_df['date'], format="%Y-%m-%d", errors='coerce')
            reddit_df = reddit_df.dropna(subset=['date'])
            
            # Load existing Reddit-compatible CSV
            if os.path.exists(news_file):
                existing_reddit_df = pd.read_csv(news_file)
                existing_reddit_df['date'] = pd.to_datetime(
                    existing_reddit_df['date'], format="%Y-%m-%d", errors='coerce'
                )
                existing_reddit_df = existing_reddit_df.dropna(subset=['date'])
            else:
                existing_reddit_df = pd.DataFrame(columns=[
                    'date', 'comment', 'subreddit', 'comment_id', 'upvotes', 'post_url',
                    'title', 'authors', 'banner_image', 'source_domain', 'topics',
                    'overall_sentiment_score', 'overall_sentiment_label', 'ticker_sentiment'
                ])
            
            # Concatenate non-empty DataFrames
            concat_dfs = [df for df in [existing_reddit_df, reddit_df] if not df.empty]
            if concat_dfs:
                updated_reddit_df = pd.concat(concat_dfs).drop_duplicates(subset=['comment_id'], keep='last')
                updated_reddit_df = updated_reddit_df.sort_values('date', ascending=True)
                updated_reddit_df['date'] = updated_reddit_df['date'].dt.strftime('%Y-%m-%d')
                updated_reddit_df.to_csv(news_file, index=False)
            else:
                print(f"Skipping Reddit-compatible CSV update for {coin_id} due to no valid data", flush=True)
        else:
            print(f"Skipping Reddit-compatible CSV update for {coin_id} due to no valid data", flush=True)
        
        # Process raw
        if raw_df is not None and not raw_df.empty:
            # Attempt to parse time_published, but preserve rows
            raw_df['time_published_original'] = raw_df['time_published']  # Keep original for debugging
            raw_df['time_published'] = pd.to_datetime(
                raw_df['time_published'], errors='coerce'
            )
            dropped_rows = raw_df[raw_df['time_published'].isna()]
            if not dropped_rows.empty:
                print(f"Dropped {len(dropped_rows)} raw rows with invalid time_published: {dropped_rows['time_published_original'].tolist()}", flush=True)
            raw_df = raw_df.dropna(subset=['time_published'])
            
            # Load existing raw CSV
            if os.path.exists(raw_news_file):
                existing_raw_df = pd.read_csv(raw_news_file)
                existing_raw_df['time_published'] = pd.to_datetime(
                    existing_raw_df['time_published'], format="%Y-%m-%d %H:%M:%S", errors='coerce'
                )
                existing_raw_df = existing_raw_df.dropna(subset=['time_published'])
            else:
                existing_raw_df = pd.DataFrame(columns=[
                    'title', 'url', 'time_published', 'authors', 'summary', 'banner_image',
                    'source', 'category_within_source', 'source_domain', 'topics',
                    'overall_sentiment_score', 'overall_sentiment_label', 'ticker_sentiment'
                ])
            
            # Concatenate non-empty DataFrames
            concat_dfs = [df for df in [existing_raw_df, raw_df] if not df.empty]
            if concat_dfs:
                updated_raw_df = pd.concat(concat_dfs).drop_duplicates(subset=['url'], keep='last')
                updated_raw_df = updated_raw_df.sort_values('time_published', ascending=True)
                updated_raw_df['time_published'] = updated_raw_df['time_published'].dt.strftime('%Y-%m-%d %H:%M:%S')
                updated_raw_df.to_csv(raw_news_file, index=False)
            else:
                print(f"Skipping raw CSV update for {coin_id} due to no valid data", flush=True)
        else:
            print(f"Skipping raw CSV update for {coin_id} due to no valid data", flush=True)
        
        total_news_counter += len(reddit_df) if reddit_df is not None and not reddit_df.empty else 0
        print(f"Updated {news_file} and {raw_news_file} with {len(reddit_df) if reddit_df is not None else 0} new news articles", flush=True)
        
        return total_news_counter
    except Exception as e:
        print(f"Error updating news for {coin_id}: {e}", flush=True)
        return total_news_counter

# Main process
def main():
    total_news_counter = 0
    
    for coin_id in COIN_MAP:
        print(f"Processing {coin_id}...", flush=True)
        total_news_counter = update_news_csv(coin_id, total_news_counter)
        time.sleep(1)  # Avoid hitting rate limits
    
    print(f"Total news articles fetched: {total_news_counter}", flush=True)

if __name__ == "__main__":
    main()