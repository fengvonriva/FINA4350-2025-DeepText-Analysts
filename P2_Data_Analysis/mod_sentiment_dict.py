import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from afinn import Afinn # AFINN sentiment scorer
from datetime import datetime # Added for timestamp
import os # Added for path manipulation

# Download NLTK punkt tokenizer (only needs to run once)
nltk.download('punkt', quiet=True)

# Source paths
comments_path = rf'P2_Data_Analysis\Sentiment_Data\LUNA_sentiment.xlsx'
price_path = rf'P2_Data_Analysis\Pricedata\LUNA_price.CSV'

def analyze_sentiment_by_criterion(
    lm_dict_path=rf'P2_Data_Analysis\Dictionaries\DICT_loughran.csv',
    comments_path=rf'P2_Data_Analysis\Sentiment_Data\LUNA_sentiment.xlsx',
    price_path=rf'P2_Data_Analysis\Pricedata\LUNA_price.CSV',
    output_path=rf'LUNA_price_x_sentiment.xlsx',
    criterion=rf'PosNegDifference'  # Default to indicate scoring method
):
    """
    Analyze sentiment of Reddit comments by calculating positive minus negative words from the Loughran-McDonald Dictionary,
    merge with price data, and save results.

    Parameters:
    - lm_dict_path (str): Path to Loughran-McDonald Dictionary CSV.
    - comments_path (str): Path to Excel file with Reddit comments.
    - price_path (str): Path to CSV file with price data.
    - output_path (str): Path to save the output Excel file.
    - criterion (str): Set to 'PosNegDifference' for positive minus negative scoring (other values ignored).

    Returns:
    - pd.DataFrame: Merged DataFrame with daily sentiment scores and price data.
    """
    # Step 1: Load Loughran-McDonald Dictionary
    lm_dict = pd.read_csv(lm_dict_path)

    # Extract positive and negative words
    positive_words = lm_dict[lm_dict['Positive'] > 0]['Word'].str.lower().tolist()
    negative_words = lm_dict[lm_dict['Negative'] > 0]['Word'].str.lower().tolist()
    print(f"Loaded {len(positive_words)} positive words and {len(negative_words)} negative words from LM Dictionary")

    # Step 2: Load the Excel file with Reddit comments
    comments_df = pd.read_excel(comments_path)

    # Step 3: Define sentiment scoring function
    def get_sentiment_score(text):
        # Tokenize the comment into words
        words = word_tokenize(text.lower())
        
        # Count positive and negative words
        pos_count = 0 # sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        # Calculate sentiment score as positive minus negative
        return pos_count - neg_count  # Positive score if pos > neg, negative if neg > pos

    # Step 4: Apply scoring to the 'Comment Text' column
    comments_df['sentiment_score'] = comments_df['Comment Text'].fillna('').apply(get_sentiment_score)

    # Step 5: Calculate daily average sentiment score
    comments_df['Comment Date'] = pd.to_datetime(comments_df['Comment Time']).dt.date
    daily_sentiment = comments_df.groupby('Comment Date')['sentiment_score'].sum().reset_index() # maybe also try sum() instead of mean().reset_index()
    daily_sentiment.rename(columns={'sentiment_score': f'avg_{criterion.lower()}_score'}, inplace=True)

    # Step 6: Load price data
    price_df = pd.read_csv(price_path)

    # Step 7: Merge daily sentiment scores with price data
    price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
    merged_df = pd.merge(daily_sentiment, price_df, left_on='Comment Date', right_on='Date', how='left')

    # Step 8: Display results
    print(f"\nMerged Data with Daily {criterion} Scores and Price Data:")
    print(merged_df)

    # Step 9: Save to a new Excel file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    base_name = os.path.splitext(output_path)[0]  # Get filename without extension
    extension = os.path.splitext(output_path)[1]  # Get extension (e.g., .xlsx)
    timestamped_output_path = rf"P2_Data_Analysis\price_x_sentiment\{base_name}_{timestamp}{extension}"
    merged_df.to_excel(timestamped_output_path, index=False)
    print(f"\nResults saved to '{timestamped_output_path}'")
    return merged_df

def analyze_sentiment_with_afinn(
    comments_path=rf'P2_Data_Analysis\Sentiment_Data\LUNA_sentiment.xlsx',
    price_path=rf'P2_Data_Analysis\Pricedata\LUNA_price.CSV',
    output_path=rf'LUNA_price_x_sentiment.xlsx'
):
    """
    Analyze sentiment of Reddit comments using the AFINN lexicon,
    merge with price data, and save results.

    Parameters:
    - afinn_path (str): Path to AFINN lexicon file (e.g., 'AFINN-en-165.txt').
    - comments_path (str): Path to Excel file with Reddit comments.
    - price_path (str): Path to CSV file with price data.
    - output_path (str): Path to save the output Excel file.

    Returns:
    - pd.DataFrame: Merged DataFrame with daily sentiment scores and price data.
    """
    # Preparation: Introduce timestamps and extension
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    base_name = os.path.splitext(output_path)[0]
    extension = ".xlsx"

    # Step 1: Initialize AFINN lexicon
    afinn = Afinn()
    print("Loaded AFINN lexicon via afinn package")

    # Step 2: Load the Excel file with Reddit comments
    comments_df = pd.read_excel(comments_path)

    # Step 3: Define sentiment scoring function
    def get_sentiment_score(text):
        # Use AFINN to score the text
        # AFINN handles tokenization internally, but we ensure empty strings return 0
        return afinn.score(text) if text.strip() else 0

    # Step 4: Apply scoring to the 'Comment Text' column
    comments_df['sentiment_score'] = comments_df['Comment Text'].fillna('').apply(get_sentiment_score)

    # Step 5: Save comments DataFrame with sentiment scores
    comments_output_path = rf"P2_Data_Analysis/Sentiment_Data/LUNA_sentiment_with_scores_{timestamp}{extension}"
    comments_df.to_excel(comments_output_path, index=False)
    print(f"\nComments with sentiment scores saved to '{comments_output_path}'")

    # Step 6: Calculate daily average sentiment score
    comments_df['Comment Date'] = pd.to_datetime(comments_df['Comment Time']).dt.date
    daily_sentiment = comments_df.groupby('Comment Date')['sentiment_score'].mean().reset_index() # consider also sum()
    daily_sentiment.rename(columns={'sentiment_score': 'avg_afinn_score'}, inplace=True)

    # Step 7: Load price data
    price_df = pd.read_csv(price_path)

    # Step 8: Merge daily sentiment scores with price data
    price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
    merged_df = pd.merge(daily_sentiment, price_df, left_on='Comment Date', right_on='Date', how='left')

    # Step 9: Display results
    print("\nMerged Data with Daily AFINN Scores and Price Data:")
    print(merged_df)

    # Step 10: Save to a new Excel file with timestamp
    timestamped_output_path = rf"P2_Data_Analysis\price_x_sentiment\{base_name}_{timestamp}{extension}"
    merged_df.to_excel(timestamped_output_path, index=False)
    print(f"\nResults saved to '{timestamped_output_path}'")

    return merged_df, comments_df

if __name__ == "__main__":
    # Example usage
    analyze_sentiment_with_afinn(comments_path, price_path)


