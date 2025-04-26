import praw
import pandas as pd
import openpyxl
import datetime
import time
import nltk

# initialize Reddit API
reddit = praw.Reddit(
    client_id='ZExuVDrnuon1q8SWA__2fw',
    client_secret='TENjvYzdpCZV2tA8gwA_8bEkyNfghg',
    user_agent='ImportanceAsleep6865',
)

# 读取帖子数据
df = pd.read_csv("BTC/BTC DATA/bitcoin_reddit_posts.csv")

comments_data = {
    "Post Title": [], 
    "Post URL": [],  # 添加帖子URL列
    "Comment Time": [], 
    "Comment Text": [], 
    "Comment Upvotes": []
}

# 获取帖子URL并添加Reddit基础URL
post_urls = ["https://reddit.com" + url for url in df["Post URL"].tolist()]

total_urls = len(post_urls)
processed_urls = 0

for url in post_urls:
    try:
        time.sleep(2)  # 减少延迟时间以加快处理速度
        submission = reddit.submission(url=url)
        submission.comments.replace_more(limit=0)  

        for comment in submission.comments.list():
            comments_data["Post Title"].append(submission.title)
            comments_data["Post URL"].append(url)
            comments_data["Comment Time"].append(
                datetime.datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            )
            comments_data["Comment Text"].append(comment.body)
            comments_data["Comment Upvotes"].append(comment.score)
        
        processed_urls += 1
        if processed_urls % 10 == 0:  # 每处理10个URL打印一次进度
            print(f"已处理 {processed_urls}/{total_urls} 个帖子")
                                                                                                                                                                                                                                                                                                                         
    except Exception as e:
        print(f"处理URL时出错 {url}\n错误信息: {str(e)}")
        time.sleep(3)  # 发生错误时稍微延长等待时间

# 保存评论数据
comments_df = pd.DataFrame(comments_data)
comments_df.to_csv("BTC/BTC DATA/bitcoin_reddit_comments.csv", index=False, encoding="utf-8")
print(f"总共保存了 {len(comments_df)} 条比特币相关评论到 bitcoin_reddit_comments.csv")
print(f"成功处理了 {processed_urls}/{total_urls} 个帖子")

nltk.download('popular')