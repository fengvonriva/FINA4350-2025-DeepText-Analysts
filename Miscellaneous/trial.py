import kagglehub
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')  # For tokenization
import ast  # For safely parsing string lists

# Step 1: Load Loughran-McDonald Dictionary
# Download from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
# Save as 'LoughranMcDonald_MasterDictionary.csv' in your working directory
lm_dict = pd.read_csv('Loughran-McDonald_Dictionary.csv')

# Extract negative words (where 'Negative' column > 0)
negative_words = lm_dict[lm_dict['Negative'] > 0]['Word'].str.lower().tolist()
print(f"Loaded {len(negative_words)} negative words from LM Dictionary")

'''# Step 2: Load the test dataset of news headlines
# Use the provided link or replace with your own file path
headlines_df = pd.read_csv('BTC.csv')
headlines = headlines_df['headline'].fillna('').tolist()'''

# Step 2: Load the test dataset from BTC.csv
# Replace 'BTC.csv' with your actual file path if different
headlines_df = pd.read_csv('BTC.csv')

# Step 3: Define sentiment scoring function
def get_lm_negativity_score(text):
    # Tokenize the headline into words
    words = word_tokenize(text.lower())
    total_words = len(words)
    
    # Count negative words
    neg_count = sum(1 for word in words if word in negative_words)
    
    # Calculate negativity score (proportion of negative words)
    if total_words == 0:
        return 0
    return neg_count / total_words  # Range: 0 to 1 (higher = more negative)

# Step 4: Parse the 'articles' column and score each headline
def parse_and_score_articles(articles_str):
    try:
        # Safely parse the string representation of the list
        articles_list = ast.literal_eval(articles_str)
        # Score each headline and return list of scores
        scores = [get_lm_negativity_score(article) for article in articles_list]
        return scores
    except (ValueError, SyntaxError):
        # Handle malformed strings gracefully
        return []

headlines_df['negativity_scores'] = headlines_df['articles'].fillna('[]').apply(parse_and_score_articles)

# Step 5: Compute average negativity score per row (optional)
headlines_df['avg_negativity_score'] = headlines_df['negativity_scores'].apply(
    lambda scores: sum(scores) / len(scores) if scores else 0
)

# Step 6: Display results
print("\nDataset with Negativity Scores:")
# Show original data with average score
print(headlines_df[['begins_at', 'symbol', 'articles', 'avg_negativity_score']])

# Optional: Expand to individual headlines for detailed view
expanded_data = []
for idx, row in headlines_df.iterrows():
    articles_list = ast.literal_eval(row['articles'])
    scores = row['negativity_scores']
    for article, score in zip(articles_list, scores):
        expanded_data.append({
            'date': row['begins_at'],
            'symbol': row['symbol'],
            'headline': article,
            'negativity_score': score
        })

expanded_df = pd.DataFrame(expanded_data)
print("\nExpanded Headlines with Individual Negativity Scores:")
print(expanded_df)

# Step 7: Save results to CSV
headlines_df.to_csv('BTC_with_negativity_scores.csv', index=False)
expanded_df.to_csv('BTC_expanded_headlines_scores.csv', index=False)
print("\nResults saved to 'BTC_with_negativity_scores.csv' and 'BTC_expanded_headlines_scores.csv'")