import praw
import pandas as pd
import openpyxl
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import re
import nltk
from nltk.corpus import stopwords
import yfinance as yf
import os
import csv
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Function to add new data to your existing CSV file
def add_news_to_csv(new_data, file_path='terra_luna_news.csv'):
    """
    Add new news items to the existing CSV file

    Parameters:
    new_data (list of dict): List of news items with required fields
    file_path (str): Path to the CSV file
    """
    # Read existing data to get column structure and existing entries
    existing_df = pd.read_csv(file_path)

    # Convert new data to DataFrame
    new_df = pd.DataFrame(new_data)

    # Ensure all required columns are present in new data
    for col in existing_df.columns:
        if col not in new_df.columns:
            new_df[col] = ''

    # Combine existing and new data
    combined_df = pd.concat([existing_df, new_df[existing_df.columns]], ignore_index=True)

    # Write back to CSV
    combined_df.to_csv(file_path, index=False)
    print(f"Successfully added {len(new_data)} new entries to {file_path}")

# Example of new data to add - you can modify this with your actual new data
new_news_items = [
    {
        "date": "02/01/2023 09:15",
        "sentiment": "{'class': 'negative', 'polarity': -0.22, 'subjectivity': 0.48}",
        "source": "CryptoNews",
        "subject": "altcoin",
        "text": "Terra Luna Classic community is still grappling with the aftermath of the collapse, with new governance proposals on the table to revitalize the ecosystem.",
        "title": "Terra Luna Classic Community Votes on New Governance Proposals to Recover Value",
        "url": "https://cryptonews.com/news/terra-luna-classic-community-votes-new-governance-proposals.htm",
        "content": ""
    },
    {
        "date": "15/01/2023 14:30",
        "sentiment": "{'class': 'positive', 'polarity': 0.15, 'subjectivity': 0.32}",
        "source": "CoinTelegraph",
        "subject": "altcoin",
        "text": "LUNC has seen a 20% price increase over the past week as new token burn mechanisms gain traction within the community.",
        "title": "Terra Luna Classic Surges 20% as Token Burn Rate Accelerates",
        "url": "https://cointelegraph.com/news/terra-luna-classic-surges-20-as-token-burn-rate-accelerates",
        "content": ""
    }
]

# Call the function to add the new data to the news CSV
print("Adding new news items to terra_luna_news.csv...")
add_news_to_csv(new_news_items)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Part 1: Reddit Data Collection
# ------------------------------

# Initialize Reddit API (fill in your credentials)
reddit = praw.Reddit(
    client_id='ZExuVDrnuon1q8SWA__2fw',    # Your client ID
    client_secret='TENjvYzdpCZV2tA8gwA_8bEkyNfghg', # Your client secret
    user_agent='ImportanceAsleep6865',    # Your user agent
)

print("Collecting Reddit posts...")

# Define subreddit and search query
subreddit = reddit.subreddit("cryptocurrency")
query = "Terra OR Luna"
search_results = subreddit.search(query, limit=2000)

# Define time range (2019-01-01 to 2022-06-30)
start_timestamp = int(datetime.datetime(2019, 1, 1).timestamp())
end_timestamp = int(datetime.datetime(2022, 6, 30).timestamp())

# Filter and save posts to DataFrame
filtered_posts = {"Post ID": [], "Title": [], "Post URL": [], "Created At": [], "Score": [], "Num Comments": []}

for post in search_results:
    if start_timestamp <= post.created_utc <= end_timestamp:
        filtered_posts["Post ID"].append(post.id)
        filtered_posts["Title"].append(post.title)
        filtered_posts["Post URL"].append(post.permalink)
        filtered_posts["Created At"].append(datetime.datetime.fromtimestamp(post.created_utc))
        filtered_posts["Score"].append(post.score)
        filtered_posts["Num Comments"].append(post.num_comments)

# Convert to Pandas DataFrame
posts_df = pd.DataFrame(filtered_posts)
print(f"Found {len(posts_df)} posts matching the criteria")
posts_df.to_csv("terra_luna_posts.csv", index=False, encoding="utf-8")

# Part 2: Collect Comments for each Post
# --------------------------------------

print("Collecting comments for each post...")

all_comments = []

# Collect comments for each post (top-level only to avoid excessive API calls)
for index, row in posts_df.iterrows():
    post_id = row['Post ID']

    submission = reddit.submission(id=post_id)
    submission.comments.replace_more(limit=0)  # Skip 'More Comments' to avoid excessive API calls

    for comment in submission.comments:
        if comment.author:  # Skip deleted comments
            all_comments.append({
                'Post ID': post_id,
                'Comment ID': comment.id,
                'Comment Text': comment.body,
                'Comment Time': datetime.datetime.fromtimestamp(comment.created_utc),
                'Comment Score': comment.score
            })

    # Print progress
    if (index + 1) % 10 == 0:
        print(f"Processed {index + 1} of {len(posts_df)} posts...")

# Convert comments to DataFrame
comments_df = pd.DataFrame(all_comments)
print(f"Collected {len(comments_df)} comments")
comments_df.to_csv("terra_luna_comments.csv", index=False, encoding="utf-8")

# Part 3: Load News Data and Process
# ---------------------------------

print("Loading and processing news data...")

# Load news data
news_df = pd.read_csv('terra_luna_news.csv')

# Convert date string to datetime
news_df['date'] = pd.to_datetime(news_df['date'], format='%d/%m/%Y %H:%M')
news_df['Date'] = news_df['date'].dt.date  # Extract date part for later merging

# Convert sentiment from string to dictionary and extract polarity
def extract_sentiment_polarity(sentiment_str):
    try:
        sentiment_dict = ast.literal_eval(sentiment_str)
        return sentiment_dict.get('polarity', 0)
    except:
        return 0

news_df['sentiment_polarity'] = news_df['sentiment'].apply(extract_sentiment_polarity)

# Combine title and text for better context
news_df['full_text'] = news_df['title'] + ' ' + news_df['text']

# Group news by date to match with daily price data
daily_news = news_df.groupby('Date').agg({
    'full_text': ' '.join,
    'sentiment_polarity': 'mean'  # Average sentiment for the day
}).reset_index()

# Part 4: Get Historical Price Data
# --------------------------------

print("Collecting historical price data for Terra Luna...")

# Try to download Luna price data (if not available, we'll create synthetic data for demo)
try:
    # Note: Terra Luna Classic ticker is 'LUNC-USD'
    luna_data = yf.download('LUNC-USD',
                           start='2019-01-01',
                           end='2022-06-30')

    # Format and save price data
    price_df = luna_data[['Close']].reset_index()
    price_df.columns = ['Date', 'Price']
    price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
    price_df.to_csv("terra_historical_price_data.csv", index=False)
    print(f"Downloaded {len(price_df)} days of price data")

except Exception as e:
    print(f"Error downloading price data: {e}")
    print("Creating synthetic price data for demonstration purposes...")

    # Create synthetic price data (based on LUNA's actual pattern - rise then crash)
    start_date = datetime.datetime(2019, 1, 1)
    end_date = datetime.datetime(2022, 6, 30)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Simulate Luna's actual price path (steep rise followed by crash)
    days = (date_range - date_range[0]).days.values
    prices = np.zeros(len(days))

    # Baseline low price for a long time
    prices[:365] = np.random.uniform(0.1, 1.0, 365)

    # Gradual rise
    rise_period = days[365:700]
    prices[365:700] = np.exp(np.linspace(0, 4, len(rise_period))) + np.random.normal(0, 0.5, len(rise_period))

    # More significant rise
    surge_period = days[700:850]
    prices[700:850] = np.exp(np.linspace(4, 6, len(surge_period))) + np.random.normal(0, 5, len(surge_period))

    # Crash
    crash_point = 850
    prices[crash_point:] = np.exp(np.linspace(6, 0, len(days) - crash_point)) + np.random.normal(0, 0.1, len(days) - crash_point)
    prices[crash_point+5:] = np.random.uniform(0.0001, 0.001, len(days) - (crash_point+5))

    # Create DataFrame
    price_df = pd.DataFrame({
        'Date': date_range.date,
        'Price': prices
    })
    price_df.to_csv("terra_historical_price_data.csv", index=False)
    print(f"Created synthetic price data with {len(price_df)} days")

# Part 5: Text Preprocessing and Feature Engineering
# -------------------------------------------------

print("Preprocessing text data...")

def clean_text(text):
    """Clean and preprocess text data"""
    if isinstance(text, str):
        # Remove URLs, special characters, numbers
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text
    return ""

# Apply cleaning to all comments
comments_df['cleaned_text'] = comments_df['Comment Text'].apply(clean_text)

# Extract comment date
comments_df['Comment Date'] = comments_df['Comment Time'].dt.date

# Group comments by date and concatenate them
daily_comments = comments_df.groupby('Comment Date')['cleaned_text'].apply(' '.join).reset_index()
daily_comments.rename(columns={'Comment Date': 'Date'}, inplace=True)

# Clean news text
daily_news['cleaned_text'] = daily_news['full_text'].apply(clean_text)

# Part 6: Merge Reddit, News, and Price Data
# ----------------------------------------

print("Merging all data sources...")

# First merge Reddit comments with price data
merged_reddit_df = pd.merge(daily_comments, price_df, on='Date', how='inner')

# Then merge news data
merged_all_df = pd.merge(merged_reddit_df, daily_news[['Date', 'cleaned_text', 'sentiment_polarity']],
                         on='Date', how='left', suffixes=('_reddit', '_news'))

# Fill NaN values for days without news (corrected for pandas warnings)
merged_all_df['cleaned_text_news'] = merged_all_df['cleaned_text_news'].fillna('')
merged_all_df['sentiment_polarity'] = merged_all_df['sentiment_polarity'].fillna(0)

# Combine Reddit and news text
merged_all_df['combined_text'] = merged_all_df['cleaned_text_reddit'] + ' ' + merged_all_df['cleaned_text_news']

# Sort by date
merged_all_df = merged_all_df.sort_values('Date')

# Save the merged data
merged_all_df.to_csv("merged_all_data.csv", index=False)

print(f"Final dataset contains {len(merged_all_df)} days with combined data")

# Check if we have enough data for meaningful analysis
if len(merged_all_df) < 30:
    print("WARNING: Not enough data for reliable deep learning analysis!")
    print("Consider expanding the data collection period or using a different approach.")
else:
    # Part 7: Prepare Data for ML (using TF-IDF instead of transformers)
    # -----------------------------------------------------------------

    print("Preparing data for machine learning model...")

    # Create a TF-IDF vectorizer for text features
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.95)

    # Prepare price data
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    prices = price_scaler.fit_transform(merged_all_df[['Price']].values)

    # Create target variable: 1 if price increases next day, 0 if decreases
    y = np.zeros(len(prices)-1)
    for i in range(len(prices)-1):
        if prices[i+1] > prices[i]:
            y[i] = 1

    # Remove last day since we don't have next day's price
    combined_text = merged_all_df['combined_text'].values[:-1]
    sentiment = merged_all_df['sentiment_polarity'].values[:-1].reshape(-1, 1)

    # Part 8: Train-Test Split (Chronological)
    # ---------------------------------------

    # Use a chronological split instead of random to preserve time sequence
    train_size = int(0.8 * len(combined_text))

    # Split the data chronologically
    X_train_text = combined_text[:train_size]
    X_test_text = combined_text[train_size:]

    X_train_sentiment = sentiment[:train_size]
    X_test_sentiment = sentiment[train_size:]

    y_train = y[:train_size]
    y_test = y[train_size:]

    print(f"Training set size: {train_size}, Test set size: {len(combined_text) - train_size}")

    # Fit and transform the training data with TF-IDF
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

    # Convert sparse matrices to dense arrays for concatenation
    X_train_tfidf_dense = X_train_tfidf.toarray()
    X_test_tfidf_dense = X_test_tfidf.toarray()

    # Combine TF-IDF features with sentiment features
    X_train = np.hstack((X_train_tfidf_dense, X_train_sentiment))
    X_test = np.hstack((X_test_tfidf_dense, X_test_sentiment))

    # Part 9: Model Training
    # --------------------

    print("Training the model...")

    # Simple Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Part 10: Evaluate the Model
    # --------------------------

    print("Evaluating model performance...")

    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Create output directory for visualizations
    os.makedirs('results', exist_ok=True)

    # Part 11: Visualize Results
    # ------------------------

    # Get dates and prices for visualization
    test_dates = merged_all_df['Date'].iloc[train_size:train_size+len(y_test)]
    test_prices = merged_all_df['Price'].iloc[train_size:train_size+len(y_test)]

    # Create DataFrames for analysis
    prediction_df = pd.DataFrame({
        'Date': test_dates,
        'Actual_Price': test_prices,
        'Predicted_Direction': y_pred,
        'Actual_Direction': y_test,
        'Prediction_Probability': y_pred_proba,
        'News_Sentiment': X_test_sentiment.flatten()
    })

    # Calculate if prediction was correct
    prediction_df['Correct_Prediction'] = prediction_df['Predicted_Direction'] == prediction_df['Actual_Direction']

    # Save predictions to CSV
    prediction_df.to_csv('results/predictions.csv', index=False)

    # Calculate performance metrics
    accuracy = prediction_df['Correct_Prediction'].mean()
    true_positives = ((prediction_df['Predicted_Direction'] == 1) & (prediction_df['Actual_Direction'] == 1)).sum()
    false_positives = ((prediction_df['Predicted_Direction'] == 1) & (prediction_df['Actual_Direction'] == 0)).sum()
    true_negatives = ((prediction_df['Predicted_Direction'] == 0) & (prediction_df['Actual_Direction'] == 0)).sum()
    false_negatives = ((prediction_df['Predicted_Direction'] == 0) & (prediction_df['Actual_Direction'] == 1)).sum()

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Print metrics
    print("\nPrediction Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot learning curve if available
    if hasattr(model, 'feature_importances_'):
        # Get feature names (top 20)
        feature_names = list(tfidf_vectorizer.get_feature_names_out()) + ['sentiment']
        feature_importance = model.feature_importances_

        # Sort feature importances and get top 20
        indices = np.argsort(feature_importance)[-20:]
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), feature_importance[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] if i < len(feature_names)-1 else 'sentiment' for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png')

    # Plot sentiment vs. price
    plt.figure(figsize=(14, 7))

    # Create twin axes
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot price and sentiment
    ax1.plot(test_dates, test_prices, 'b-', label='Price')
    ax2.plot(test_dates, X_test_sentiment, 'r-', label='News Sentiment')

    # Customize plot
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)', color='b')
    ax2.set_ylabel('Sentiment', color='r')

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('TerraLuna Price vs. News Sentiment')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/price_vs_sentiment.png')

    # Part 12: Word Importance Analysis
    # ------------------------------

    # For TF-IDF model, we can look at the most important features
    from collections import Counter

    # SIMPLIFIED APPROACH: Work with the prediction_df directly
    # This dataframe already contains all the information we need

    # Get all texts from the test period (already stored in X_test_text)
    test_texts = X_test_text

    # Create a list of booleans for correct predictions
    correct_predictions = prediction_df['Correct_Prediction'].values

    # Get the texts that correspond to correct predictions
    correct_texts = [text for text, is_correct in zip(test_texts, correct_predictions) if is_correct]

    # If no correct predictions, use all texts
    if len(correct_texts) == 0:
        print("Warning: No correct predictions found. Using all test texts instead.")
        correct_texts = test_texts

    # Get words from correctly predicted texts
    correct_words = []
    for text in correct_texts:
        words = text.split()
        correct_words.extend(words)

    # Get most common words
    word_counts = Counter(correct_words)
    top_words = word_counts.most_common(20)

    # Create DataFrame of top words
    top_words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    top_words_df.to_csv('results/influential_words.csv', index=False)

    # Visualize top words
    plt.figure(figsize=(12, 6))
    plt.barh(
        [word for word, count in top_words],
        [count for word, count in top_words]
    )
    plt.title('Most Frequent Words in Correctly Predicted Texts')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.tight_layout()
    plt.savefig('results/influential_words.png')

    # Part 13: Save the Model and Vectorizer
    # ------------------------------------

    # Import pickle for model saving
    import pickle

    # Save the model
    with open('results/terra_luna_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Save the vectorizer
    with open('results/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    print("\nAnalysis complete! Results saved to the 'results' directory.")
    print("\nSummary of files generated:")
    print("- terra_luna_posts.csv: Reddit posts about Terra/Luna")
    print("- terra_luna_comments.csv: Comments from the posts")
    print("- terra_historical_price_data.csv: Historical price data")
    print("- merged_all_data.csv: Combined Reddit, news and price data")
    print("- results/predictions.csv: Model predictions and actual values")
    print("- results/feature_importance.png: Feature importances from the model")
    print("- results/price_vs_sentiment.png: Price vs news sentiment visualization")
    print("- results/influential_words.png: Most frequent words in correctly predicted samples")
    print("- results/terra_luna_model.pkl: Saved model")
    print("- results/tfidf_vectorizer.pkl: Saved vectorizer for future predictions")