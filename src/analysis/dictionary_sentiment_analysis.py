import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import os

# download necessary NLTK data  # English comments and English output
nltk.download('punkt', quiet=True)

def analyze_btc_sentiment(
    lm_dict_path='LoughranMcDonald_Dictionary.csv',
    comments_path='BTC/BTC DATA/bitcoin_reddit_comments.csv',
    output_path='BTC/BTC DATA/btc_comments_with_sentiment.csv'
):
    """
    Analyze the sentiment of Bitcoin-related comments
    
    Parameters:
    - lm_dict_path: Loughran-McDonald dictionary path
    - comments_path: Bitcoin comment data path
    - output_path: Output file path
    """
    
    # load Loughran-McDonald dictionary
    print("Loading sentiment dictionary...")
    lm_dict = pd.read_csv("D:/sjtucyx iii/2425/FINA4350/data/Loughran-McDonald_Dictionary.csv")
    
    # extract various sentiment words
    negative_words = set(lm_dict[lm_dict['Negative'] > 0]['Word'].str.lower().tolist())
    positive_words = set(lm_dict[lm_dict['Positive'] > 0]['Word'].str.lower().tolist())
    uncertainty_words = set(lm_dict[lm_dict['Uncertainty'] > 0]['Word'].str.lower().tolist())
    
    print(f"Loaded {len(negative_words)} negative words, {len(positive_words)} positive words, {len(uncertainty_words)} uncertainty words")

    # load comment data
    print("Loading comment data...")
    comments_df = pd.read_csv(comments_path)
    total_comments = len(comments_df)
    print(f"Loaded {total_comments} comments")

    # define sentiment analysis function
    def get_sentiment_scores(text):
        if pd.isna(text):
            return 0, 0, 0
        
        # tokenize
        words = word_tokenize(str(text).lower())
        total_words = len(words)
        
        if total_words == 0:
            return 0, 0, 0
            
        # calculate the number of various sentiment words
        negative_count = sum(1 for word in words if word in negative_words)
        positive_count = sum(1 for word in words if word in positive_words)
        uncertainty_count = sum(1 for word in words if word in uncertainty_words)
        
        # calculate the sentiment score (proportion)
        return (
            negative_count / total_words,
            positive_count / total_words,
            uncertainty_count / total_words
        )

    # apply sentiment analysis
    print("Applying sentiment analysis...")
    
    # use list comprehension to improve efficiency
    sentiment_scores = [get_sentiment_scores(text) for text in comments_df['Comment Text']]
    
    # add results to DataFrame
    comments_df['negative_score'] = [score[0] for score in sentiment_scores]
    comments_df['positive_score'] = [score[1] for score in sentiment_scores]
    comments_df['uncertainty_score'] = [score[2] for score in sentiment_scores]
    
    # calculate the comprehensive sentiment score (positive - negative)
    comments_df['sentiment_score'] = comments_df['positive_score'] - comments_df['negative_score']

    # add sentiment label
    def get_sentiment_label(row):
        if row['sentiment_score'] > 0:
            return '正面'
        elif row['sentiment_score'] < 0:
            return '负面'
        else:
            return '中性'

    comments_df['sentiment_label'] = comments_df.apply(get_sentiment_label, axis=1)

    # save results
    print("Saving analysis results...")
    comments_df.to_csv(output_path, index=False, encoding='utf-8')
    
    # 计算加权每日情感得分
    def calculate_weighted_sentiment(group):
        # use upvotes as weights, if upvotes is negative, take the absolute value plus 1 (to avoid negative weights)
        weights = group['Comment Upvotes'].apply(lambda x: abs(x) + 1 if x < 0 else x + 1)
        
        # calculate the weighted average sentiment score
        weighted_sentiment = (group['sentiment_score'] * weights).sum() / weights.sum()
        
        return pd.Series({
            'Weighted_Sentiment': weighted_sentiment,
            'Total_Comments': len(group),
            'Total_Upvotes': weights.sum() - len(group),  # minus the 1 we added
            'Avg_Upvotes': (weights.sum() - len(group)) / len(group),
            'Max_Upvotes': weights.max() - 1,
            'Min_Upvotes': weights.min() - 1
        })

    # calculate the daily weighted sentiment score
    comments_df['Comment Time'] = pd.to_datetime(comments_df['Comment Time'])
    daily_sentiment = comments_df.groupby(comments_df['Comment Time'].dt.date).apply(calculate_weighted_sentiment).reset_index()
    daily_sentiment.rename(columns={'Comment Time': 'Date'}, inplace=True)  # rename columns
    
    # add simple average sentiment score for comparison
    daily_simple_avg = comments_df.groupby(comments_df['Comment Time'].dt.date)['sentiment_score'].mean().reset_index()
    daily_simple_avg.columns = ['Date', 'Simple_Avg_Sentiment']
    
    # merge weighted and simple average results
    daily_sentiment = daily_sentiment.merge(daily_simple_avg, on='Date')
    
    # save daily sentiment statistics
    daily_sentiment.to_csv('BTC/BTC DATA/daily_btc_sentiment.csv', index=False)
    print("\nDaily sentiment statistics saved to 'daily_btc_sentiment.csv'")
    
    # output some basic statistics
    print("\nData statistics summary:")
    print(f"Analysis time range: {daily_sentiment['Date'].min()} to {daily_sentiment['Date'].max()}")
    print(f"Total days: {len(daily_sentiment)}")
    print(f"Average daily comments: {daily_sentiment['Total_Comments'].mean():.2f}")
    
    return comments_df, daily_sentiment 

if __name__ == "__main__":
    # run sentiment analysis
    comments_df, daily_sentiment = analyze_btc_sentiment() 