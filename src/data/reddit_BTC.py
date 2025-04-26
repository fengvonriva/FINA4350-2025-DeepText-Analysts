import praw
import pandas as pd
import openpyxl
import datetime

# initialize Reddit API
reddit = praw.Reddit(
    client_id='ZExuVDrnuon1q8SWA__2fw',
    client_secret='TENjvYzdpCZV2tA8gwA_8bEkyNfghg',
    user_agent='ImportanceAsleep6865',
)



subreddit = reddit.subreddit("cryptocurrency")

# 使用多个关键词搜索
search_terms = ["BTC", "Bitcoin", "bitcoin", "BITCOIN"]
filtered_posts = {"Title": [], "Post URL": [], "Created At": []}

# 计算最近两年的时间范围
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=365*2)
start_timestamp = int(start_date.timestamp())
end_timestamp = int(end_date.timestamp())

# 对每个关键词进行搜索
for term in search_terms:
    search_results = subreddit.search(term, limit=2000)
    
    for post in search_results:
        if start_timestamp <= post.created_utc <= end_timestamp:
            # 避免重复帖子
            if post.permalink not in filtered_posts["Post URL"]:
                filtered_posts["Title"].append(post.title)
                filtered_posts["Post URL"].append(post.permalink)
                filtered_posts["Created At"].append(datetime.datetime.fromtimestamp(post.created_utc))

# convert to Pandas DataFrame
df = pd.DataFrame(filtered_posts)
print(df.head())
df.to_csv("BTC/BTC DATA/bitcoin_reddit_posts.csv", index=False, encoding="utf-8")
print(f"There are {len(df)} unique posts related to Bitcoin saved to bitcoin_reddit_posts.csv")

