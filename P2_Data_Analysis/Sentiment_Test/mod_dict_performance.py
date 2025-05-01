import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from afinn import Afinn
import json
import string
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize AFINN
afinn = Afinn()

# File paths
training_data_path = "P2_Data_Analysis/Sentiment_Test/cryptonews_sentiment_training.csv"
loughran_dict_path = "P2_Data_Analysis/Dictionaries/DICT_loughran.csv"
preprocessed_output_path = "P2_Data_Analysis/Sentiment_Test/preprocessed_cryptonews.csv"
performance_output_path = "P2_Data_Analysis/Sentiment_Test/DICT_performance.csv"

# Load datasets with encoding specification
try:
    df = pd.read_csv(training_data_path, encoding='utf-8')
except UnicodeDecodeError:
    print("UTF-8 decoding failed, trying 'latin1' encoding...")
    df = pd.read_csv(training_data_path, encoding='latin1')

loughran_df = pd.read_csv(loughran_dict_path)

# Extract positive and negative words from Loughran-McDonald
loughran_positive = set(loughran_df[loughran_df['Positive'] != 0]['Word'].str.lower())
loughran_negative = set(loughran_df[loughran_df['Negative'] != 0]['Word'].str.lower())

# Get NLTK stopwords, but retain Loughran-McDonald sentiment words
stop_words = set(stopwords.words('english'))
loughran_sentiment_words = loughran_positive.union(loughran_negative)
stop_words = stop_words - loughran_sentiment_words

# Preprocess function
def preprocess_text(text):
    # Handle non-string inputs (e.g., NaN, float)
    if not isinstance(text, str):
        return []
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Handle missing or non-string values in 'title' column
df['title'] = df['title'].fillna('').astype(str)

# Apply preprocessing to titles
df['preprocessed_title'] = df['title'].apply(preprocess_text)

# Save preprocessed data
preprocessed_df = df[['date', 'sentiment', 'source', 'subject', 'text', 'title', 'url', 'preprocessed_title']]
preprocessed_df.to_csv(preprocessed_output_path, index=False)

# Loughran-McDonald scoring
def loughran_score(tokens):
    pos_count = sum(1 for token in tokens if token in loughran_positive)
    neg_count = sum(1 for token in tokens if token in loughran_negative)
    score = pos_count - neg_count
    sentiment = 'positive' if score > 0 else 'negative' if score < 0 else 'neutral'
    return score, sentiment

# AFINN scoring
def afinn_score(tokens):
    score = sum(afinn.score(token) for token in tokens)
    sentiment = 'positive' if score > 0 else 'negative' if score < 0 else 'neutral'
    return score, sentiment

# Apply scoring
df[['Loughran_Score', 'Loughran_sentiment']] = df['preprocessed_title'].apply(loughran_score).apply(pd.Series)
df[['AFINN_score', 'AFINN_sentiment']] = df['preprocessed_title'].apply(afinn_score).apply(pd.Series)

# Extract actual sentiment from JSON-like string
def extract_sentiment(sentiment_str):
    try:
        sentiment_dict = json.loads(sentiment_str.replace("'", "\""))
        return sentiment_dict['class']
    except:
        return 'neutral'  # Fallback for malformed JSON

df['actual_sentiment'] = df['sentiment'].apply(extract_sentiment)

# Create output DataFrame
output_df = df[['date', 'sentiment', 'source', 'subject', 'text', 'title', 'url',
                'Loughran_Score', 'AFINN_score', 'Loughran_sentiment', 'AFINN_sentiment', 'actual_sentiment']]

# Save performance output
output_df.to_csv(performance_output_path, index=False)

# Compute accuracy statistics
def compute_metrics(actual, predicted, model_name):
    accuracy = accuracy_score(actual, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average='weighted', zero_division=0)
    cm = confusion_matrix(actual, predicted, labels=['positive', 'negative', 'neutral'])
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['positive', 'negative', 'neutral'], yticklabels=['positive', 'negative', 'neutral'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'P2_Data_Analysis/Sentimentdata/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()

# Compute metrics for both dictionaries
compute_metrics(df['actual_sentiment'], df['Loughran_sentiment'], "Loughran-McDonald")
compute_metrics(df['actual_sentiment'], df['AFINN_sentiment'], "AFINN")