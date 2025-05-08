import requests
import pandas as pd
import os
from datetime import datetime, date
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

# Terra Luna ticker
TERRA_LUNA_TICKER = "CRYPTO:LUNA"

# Ensure directory exists
os.makedirs(SENTIMENT_DIR, exist_ok=True)

# Validate and extract date from time_published
def extract_date(time_published):
    try:
        # Expect format like YYYYMMDDTHHMMSS or YYYY-MM-DDTHH:MM:SS
        if not isinstance(time_published, str):
            print(f"Invalid time_published type: {time_published}", flush=True)
            return None
        # Match YYYYMMDD or YYYY-MM-DD
        match = re.match(r"^(\d{4})(\d{2})(\d{2})T|\d{4}-\d{2}-\d{2}T", time_published)
        if not match:
            print(f"Invalid time_published format: {time_published}", flush=True)
            return None
        # Extract YYYY-MM-DD
        date_str = time_published[:10]
        if len(date_str) != 10 or date_str[4] != '-' and date_str[7] != '-':
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
def fetch_news(ticker, time_from, sort="LATEST"):
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={time_from}&limit=1000&sort={sort}&apikey={API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if "feed" not in data or not data["feed"]:
            print(f"No news found for {ticker} from {time_from}", flush=True)
            return None, None
        
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
        
        if not news_items:
            print(f"No valid news articles after date filtering for {ticker} from {time_from}", flush=True)
            return None, None
        
        reddit_df = pd.DataFrame(news_items)
        
        # Raw DataFrame
        raw_items = []
        for article in data["feed"]:
            date_str = extract_date(article["time_published"])
            if date_str is None:
                continue
            article_copy = article.copy()
            article_copy["time_published"] = date_str + article["time_published"][10:]  # Preserve time
            raw_items.append(article_copy)
        
        if not raw_items:
            print(f"No valid raw news articles after date filtering for {ticker} from {time_from}", flush=True)
            return None, None
        
        raw_df = pd.DataFrame(raw_items)
        
        print(f"Fetched {len(reddit_df)} news articles for {ticker} from {time_from}", flush=True)
        return reddit_df, raw_df
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}", flush=True)
        return None, None

# Main process
def main():
    news_file = os.path.join(SENTIMENT_DIR, "terra-luna_news.csv")
    raw_news_file = os.path.join(SENTIMENT_DIR, "terra-luna_news_raw.csv")
    
    # Define periods
    periods = [
        {
            "time_from": "20220101T0000",
            "sort": "EARLIEST"
        }
    ]
    
    all_reddit_news = []
    all_raw_news = []
    
    for period in periods:
        time_from = period["time_from"]
        sort = period["sort"]
        print(f"Fetching news for Terra Luna from {time_from}", flush=True)
        reddit_df, raw_df = fetch_news(TERRA_LUNA_TICKER, time_from, sort)
        if reddit_df is not None and raw_df is not None:
            all_reddit_news.append(reddit_df)
            all_raw_news.append(raw_df)
        time.sleep(1)  # Avoid hitting rate limits
    
    if not all_reddit_news:
        print(f"No news found for Terra Luna", flush=True)
        return
    
    # Combine and sort Reddit-compatible
    combined_reddit_df = pd.concat(all_reddit_news, ignore_index=True)
    combined_reddit_df['date'] = pd.to_datetime(combined_reddit_df['date'], format="%Y-%m-%d", errors='coerce')
    combined_reddit_df = combined_reddit_df.dropna(subset=['date'])  # Drop rows with invalid dates
    combined_reddit_df = combined_reddit_df.drop_duplicates(subset=['comment_id'], keep='last')
    combined_reddit_df = combined_reddit_df.sort_values('date', ascending=True)
    combined_reddit_df['date'] = combined_reddit_df['date'].dt.strftime('%Y-%m-%d')
    
    # Combine and sort raw
    combined_raw_df = pd.concat(all_raw_news, ignore_index=True)
    combined_raw_df['time_published'] = pd.to_datetime(
        combined_raw_df['time_published'], format="%Y-%m-%dT%H%M%S", errors='coerce'
    )
    combined_raw_df = combined_raw_df.dropna(subset=['time_published'])  # Drop rows with invalid dates
    combined_raw_df = combined_raw_df.drop_duplicates(subset=['url'], keep='last')
    combined_raw_df = combined_raw_df.sort_values('time_published', ascending=True)
    combined_raw_df['time_published'] = combined_raw_df['time_published'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to CSVs
    combined_reddit_df.to_csv(news_file, index=False)
    combined_raw_df.to_csv(raw_news_file, index=False)
    print(f"Saved {news_file} and {raw_news_file} with {len(combined_reddit_df)} news articles", flush=True)

if __name__ == "__main__":
    main()