import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK punkt tokenizer (only needs to run once)
nltk.download('punkt', quiet=True)

def analyze_sentiment_by_criterion(
    lm_dict_path='LoughranMcDonald_Dictionary.csv',
    comments_path='filtered_reddit_selected_rows.xlsx',
    price_path='terra-historical-day-data-all-tokeninsight.csv',
    output_path='daily_sentiment_and_prices.xlsx',
    criterion='Negative'
):
    """
    Analyze sentiment of Reddit comments using a specified criterion from the Loughran-McDonald Dictionary,
    merge with price data, and save results.

    Parameters:
    - lm_dict_path (str): Path to Loughran-McDonald Dictionary CSV.
    - comments_path (str): Path to Excel file with Reddit comments.
    - price_path (str): Path to CSV file with price data.
    - output_path (str): Path to save the output Excel file.
    - criterion (str): Sentiment criterion to use (e.g., 'Negative', 'Positive', 'Uncertainty', 'Litigious').

    Returns:
    - pd.DataFrame: Merged DataFrame with daily sentiment scores and price data.
    """
    # Step 1: Load Loughran-McDonald Dictionary
    lm_dict = pd.read_csv(lm_dict_path)

    # Validate criterion
    valid_criteria = ['Negative', 'Positive', 'Uncertainty', 'Litigious']
    if criterion not in valid_criteria:
        raise ValueError(f"Criterion must be one of {valid_criteria}")

    # Extract words for the specified criterion (where criterion column > 0)
    sentiment_words = lm_dict[lm_dict[criterion] > 0]['Word'].str.lower().tolist()
    print(f"Loaded {len(sentiment_words)} {criterion.lower()} words from LM Dictionary")

    # Step 2: Load the Excel file with Reddit comments
    comments_df = pd.read_excel(comments_path)

    # Step 3: Define sentiment scoring function
    def get_sentiment_score(text):
        # Tokenize the comment into words
        words = word_tokenize(text.lower())
        total_words = len(words)

        # Count criterion-specific words
        sentiment_count = sum(1 for word in words if word in sentiment_words)

        # Calculate sentiment score (proportion of criterion-specific words)
        if total_words == 0:
            return 0
        return sentiment_count / total_words  # Range: 0 to 1 (higher = more of the criterion)

    # Step 4: Apply scoring to the 'Comment Text' column
    comments_df['sentiment_score'] = comments_df['Comment Text'].fillna('').apply(get_sentiment_score)

    # Step 5: Calculate daily average sentiment score
    comments_df['Comment Date'] = pd.to_datetime(comments_df['Comment Time']).dt.date
    daily_sentiment = comments_df.groupby('Comment Date')['sentiment_score'].mean().reset_index()
    daily_sentiment.rename(columns={'sentiment_score': f'avg_{criterion.lower()}_score'}, inplace=True)

    # Step 6: Load price data
    price_df = pd.read_csv(price_path)

    # Step 7: Merge daily sentiment scores with price data
    price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
    merged_df = pd.merge(daily_sentiment, price_df, left_on='Comment Date', right_on='Date', how='left')

    # Step 8: Display results
    print(f"\nMerged Data with Daily {criterion} Scores and Price Data:")
    print(merged_df)

    # Step 9: Save to a new Excel file
    merged_df.to_excel(output_path, index=False)
    print(f"\nResults saved to '{output_path}'")

    return merged_df

if __name__ == "__main__":
    # Example usage
    analyze_sentiment_by_criterion(criterion='Negative')