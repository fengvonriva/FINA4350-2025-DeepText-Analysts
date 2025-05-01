import pandas as pd

# Load Loughran McDonald dictionary
DICT_PATH = 'P2_Data_Analysis/Dictionaries/DICT_loughran.csv'

try:
    dict_df = pd.read_csv(DICT_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Dictionary file {DICT_PATH} not found.")

# Validate dictionary columns
required_columns = {'Word', 'Positive', 'Negative'}
if not required_columns.issubset(dict_df.columns):
    raise ValueError(f"Dictionary must contain {required_columns} columns.")

# Create word lists
dict_df['Word'] = dict_df['Word'].str.lower()
positive_words = set(dict_df[dict_df['Positive'] != 0]['Word'])
negative_words = set(dict_df[dict_df['Negative'] != 0]['Word'])

def analyze_loughran_mcdonald(text):
    """
    Analyze sentiment of a preprocessed string using Loughran McDonald dictionary.
    Args:
        text (str): Preprocessed string (lowercase, no stopwords).
    Returns:
        int: Sentiment score (number of positive words - number of negative words).
    """
    words = text.split()  # Assumes preprocessed text is space-separated
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    return positive_count - negative_count