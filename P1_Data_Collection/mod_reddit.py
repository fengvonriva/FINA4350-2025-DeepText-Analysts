import praw
import pandas as pd
import openpyxl
import datetime
import time

# initialize Reddit API
reddit = praw.Reddit(
    client_id='ZExuVDrnuon1q8SWA__2fw',
    client_secret='TENjvYzdpCZV2tA8gwA_8bEkyNfghg',
    user_agent='ImportanceAsleep6865',
)


subreddit = reddit.subreddit("cryptocurrency")
query = "Terra OR Luna"
search_results = subreddit.search(query, limit=2000)

# time range(2021.1 - 2022.6)
start_timestamp = int(datetime.datetime(2019, 1, 1).timestamp())  # 2019-01-01
end_timestamp = int(datetime.datetime(2022, 6, 30).timestamp())  # 2022-06-30

# filter and save to DataFrame
filtered_posts = {"Title": [], "Post URL": [], "Created At": []}

for post in search_results:
    if start_timestamp <= post.created_utc <= end_timestamp:
        filtered_posts["Title"].append(post.title)
        filtered_posts["Post URL"].append(post.permalink)
        filtered_posts["Created At"].append(datetime.datetime.fromtimestamp(post.created_utc))

# convert to Pandas DataFrame
df = pd.DataFrame(filtered_posts)
print(df.head())
df.to_csv("terra_luna_posts.csv", index=False, encoding="utf-8")

print(f" There are {len(df)} of post satisified the requirement and saved to terra_luna_posts.csv")


# initialize Reddit API
reddit = praw.Reddit(
    client_id='ZExuVDrnuon1q8SWA__2fw',
    client_secret='TENjvYzdpCZV2tA8gwA_8bEkyNfghg',
    user_agent='ImportanceAsleep6865',
)

# open the terra_luna_posts.csv file and extract the data
df = pd.read_csv("terra_luna_posts.csv")

comments_data = {"Post Title": [], "Comment Time": [], "Comment Text": []}
# extract the post URLs
post_urls = df["Post URL"].tolist()
for url in post_urls:
    try:
        time.sleep(3)
        submission = reddit.submission(url=url)
        submission.comments.replace_more(limit=0)

        for comment in submission.comments.list():
            comments_data["Post Title"].append(submission.title)
            comments_data["Comment Time"].append(
                datetime.datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            )
            comments_data["Comment Text"].append(comment.body)

    except Exception as e:
        print(f"skip {url}, the error is: {e}")
        time.sleep(5)

#
comments_df = pd.DataFrame(comments_data)
comments_df.to_csv("gp/reddit_comments.csv", index=False, encoding="utf-8")
print(comments_df.head())