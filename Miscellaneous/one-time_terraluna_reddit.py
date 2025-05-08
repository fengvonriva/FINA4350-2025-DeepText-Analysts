import praw
import pandas as pd
import os
from datetime import datetime, timezone
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
SENTIMENT_DIR = "P2_Data_Analysis/Sentimentdata"

# Ensure directory exists
os.makedirs(SENTIMENT_DIR, exist_ok=True)

# Terra Luna terms for filtering
TERRA_LUNA_TERMS = ["terra", "terra luna", "luna", "LUNA"]

# Fetch comments for posts
def fetch_comments_for_posts(posts_df):
    try:
        comments = []
        post_urls = ["https://reddit.com" + url for url in posts_df["Post URL"].tolist()]
        total_urls = len(post_urls)
        processed_urls = 0
        
        for url in post_urls:
            try:
                time.sleep(2)
                print(f"Fetching comments for post {url}", flush=True)
                submission = reddit.submission(url=url)
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list():
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
            print(f"No comments found for posts", flush=True)
            return None
        
        df = pd.DataFrame(comments)
        print(f"Fetched {len(df)} total comments", flush=True)
        return df
    except Exception as e:
        print(f"Error fetching comments: {e}", flush=True)
        return None

# Filter comments for Terra Luna
def filter_comments_for_terra_luna(df):
    if df is None or df.empty:
        return None
    
    pattern = rf"\b({'|'.join(re.escape(term) for term in TERRA_LUNA_TERMS)})\b"
    mask = df['comment'].str.contains(pattern, case=False, na=False)
    filtered_df = df[mask].copy()
    
    if filtered_df.empty:
        return None
    
    return filtered_df[['date', 'comment', 'subreddit', 'comment_id', 'upvotes', 'post_url']]

# Main process
def main():
    # Test authentication
    print("Testing authentication...")
    try:
        print(f"Authenticated as: {reddit.user.me()}", flush=True)
    except Exception as e:
        print(f"Authentication failed: {e}", flush=True)
        exit()
    
    # Paths
    posts_file = os.path.join(SENTIMENT_DIR, "terra-luna_posts.csv")
    comments_file = os.path.join(SENTIMENT_DIR, "terra-luna_comments.csv")
    
    # Load and sort posts CSV
    if not os.path.exists(posts_file):
        print(f"Posts file {posts_file} not found", flush=True)
        return
    
    posts_df = pd.read_csv(posts_file)
    posts_df['Created At'] = pd.to_datetime(posts_df['Created At'])
    posts_df = posts_df.sort_values('Created At', ascending=True)
    posts_df['Created At'] = posts_df['Created At'].dt.strftime('%Y-%m-%d')
    posts_df.to_csv(posts_file, index=False)
    print(f"Sorted {posts_file} by Created At", flush=True)
    
    # Fetch comments
    comments_df = fetch_comments_for_posts(posts_df)
    if comments_df is None:
        print(f"No comments found for Terra Luna posts", flush=True)
        return
    
    # Filter comments
    filtered_comments = filter_comments_for_terra_luna(comments_df)
    if filtered_comments is None:
        print(f"No comments found containing Terra Luna terms", flush=True)
        return
    
    # Sort comments by date
    filtered_comments['date'] = pd.to_datetime(filtered_comments['date'])
    filtered_comments = filtered_comments.sort_values('date', ascending=True)
    filtered_comments['date'] = filtered_comments['date'].dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    filtered_comments.to_csv(comments_file, index=False)
    print(f"Saved {comments_file} with {len(filtered_comments)} comments", flush=True)

if __name__ == "__main__":
    main()