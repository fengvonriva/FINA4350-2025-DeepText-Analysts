import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import ast

nltk.download('punkt')  # For tokenization

# Step 1: Load Loughran-McDonald Dictionary
# Download from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
# Save as 'LoughranMcDonald_MasterDictionary.csv' in your working directory
lm_dict = pd.read_csv('Loughran-McDonald_Dictionary.csv')

# Extract negative words (where 'Negative' column > 0)
negative_words = lm_dict[lm_dict['Negative'] > 0]['Word'].str.lower().tolist()
print(f"Loaded {len(negative_words)} negative words from LM Dictionary")

# Step 2: Load the Excel file with Reddit comments
# Replace 'reddit_comments.xlsx' with your actual file path
comments_df = pd.read_excel('filtered_reddit_selected_rows.xlsx')

# Step 3: Define sentiment scoring function
def get_lm_negativity_score(text):
    # Tokenize the comment into words
    words = word_tokenize(text.lower())
    total_words = len(words)

    # Count negative words
    neg_count = sum(1 for word in words if word in negative_words)

    # Calculate negativity score (proportion of negative words)
    if total_words == 0:
        return 0
    return neg_count / total_words  # Range: 0 to 1 (higher = more negative)

# Step 4: Apply scoring to the 'Comment Text' column
comments_df['negativity_score'] = comments_df['Comment Text'].fillna('').apply(get_lm_negativity_score)

# Step 5: Calculate daily average negativity score
# Convert Comment Time to datetime and extract date only
comments_df['Comment Date'] = pd.to_datetime(comments_df['Comment Time']).dt.date
daily_negativity = comments_df.groupby('Comment Date')['negativity_score'].mean().reset_index()
daily_negativity.rename(columns={'negativity_score': 'avg_negativity_score'}, inplace=True)

# Step 6: Load price data
# Replace 'terra-historical-day-data-all-tokeninsight.csv' with your actual file path
price_df = pd.read_csv('terra-historical-day-data-all-tokeninsight.csv')

# Step 7: Merge daily negativity scores with price data
# Ensure date formats match (convert price Date to date)
price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
merged_df = pd.merge(daily_negativity, price_df, left_on='Comment Date', right_on='Date', how='left')

# Step 8: Display results
print("\nMerged Data with Daily Negativity Scores and Price Data:")
print(merged_df)

# Step 9: Save to a new Excel file
merged_df.to_excel('daily_negativity_and_prices.xlsx', index=False)
print("\nResults saved to 'daily_negativity_and_prices.xlsx'")
