import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import re
import nltk
from nltk.corpus import stopwords
import yfinance as yf
import os
import datetime
import ast
from sklearn.metrics import classification_report, confusion_matrix

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

print("Starting Bitcoin sentiment analysis with transformers...")

# Initialize the transformer model and tokenizer
# Using a smaller, efficient model
transformer_model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
max_len = 128  # Maximum sequence length (shorter than BERT's 512 for efficiency)

# Part 1: Load Bitcoin Tweet Data
# ------------------------------
print("Loading Bitcoin tweets dataset...")

# Process the file in chunks to handle memory constraints
chunk_size = 50000  # Adjust based on your system's memory
chunks = []
total_rows = 0

print("Reading Bitcoin tweets in chunks...")

try:
    # Create an iterator that processes the file in chunks
    chunk_iterator = pd.read_csv('Bitcoin_tweets.csv',
                             chunksize=chunk_size,
                             on_bad_lines='skip',  # Skip problematic rows
                             quoting=3,            # QUOTE_NONE - don't do special processing for quotes
                             escapechar='\\',      # Use backslash as escape character
                             low_memory=True)      # Use less memory

    # Process each chunk
    for i, chunk in enumerate(chunk_iterator):
        # Add to our list of chunks
        chunks.append(chunk)
        total_rows += len(chunk)

        # Print progress
        print(f"Processed chunk {i+1} with {len(chunk)} rows. Total: {total_rows} rows")

        # Optional: limit to first X chunks for testing (comment out for full data)
        # if i >= 5:  # Limit to first 5 chunks
        #    break

    # Combine all chunks
    bitcoin_tweets_df = pd.concat(chunks, ignore_index=True)
    print(f"Successfully loaded {len(bitcoin_tweets_df)} tweets")

except Exception as e:
    print(f"Error during processing: {e}")

    # If we have some data, still create the dataframe
    if chunks:
        bitcoin_tweets_df = pd.concat(chunks, ignore_index=True)
        print(f"Partial data loaded: {len(bitcoin_tweets_df)} tweets")
    else:
        print("No data was successfully loaded")
        # Create empty dataframe or handle no-data case
        bitcoin_tweets_df = pd.DataFrame()

print(f"Loaded dataset with {len(bitcoin_tweets_df)} tweets")
print(f"Columns in the dataset: {bitcoin_tweets_df.columns.tolist()}")

# Part 2: Data Preprocessing
# -------------------------
print("Preprocessing tweet data...")

# First, check if there are non-date values in the date column
print("Sample of date values:", bitcoin_tweets_df['date'].head(10).tolist())
print("Number of 'False' values:", (bitcoin_tweets_df['date'] == 'False').sum())

# Convert date to datetime with error handling
bitcoin_tweets_df['date_parsed'] = pd.to_datetime(
    bitcoin_tweets_df['date'],
    errors='coerce',  # This will set invalid dates to NaT (Not a Time)
    format='mixed'    # Try to infer the format for each value
)

# Now you can check how many dates couldn't be parsed
print("Number of invalid dates:", bitcoin_tweets_df['date_parsed'].isna().sum())

# Optionally, filter out rows with invalid dates
bitcoin_tweets_df = bitcoin_tweets_df.dropna(subset=['date_parsed'])
print(f"Remaining rows after removing invalid dates: {len(bitcoin_tweets_df)}")

# Extract date only (without time) for grouping
bitcoin_tweets_df['date_only'] = bitcoin_tweets_df['date_parsed'].dt.date

# If you need to keep using the 'date' column name instead of 'date_parsed'
bitcoin_tweets_df['date'] = bitcoin_tweets_df['date_parsed']

# Clean and preprocess text data
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

# Apply text cleaning
bitcoin_tweets_df['cleaned_text'] = bitcoin_tweets_df['text'].apply(clean_text)

# Part 3: Get Historical Bitcoin Price Data
# ---------------------------------------
print("Collecting historical Bitcoin price data...")

# Define date range based on the tweets dataset
start_date = bitcoin_tweets_df['date'].min().date()
end_date = bitcoin_tweets_df['date'].max().date()

# Add some buffer days
start_date = start_date - datetime.timedelta(days=7)
end_date = end_date + datetime.timedelta(days=7)

print(f"Fetching Bitcoin prices from {start_date} to {end_date}")

# Download Bitcoin price data
try:
    bitcoin_data = yf.download('BTC-USD',
                             start=start_date,
                             end=end_date)

    # Format and save price data
    price_df = bitcoin_data[['Close']].reset_index()
    price_df.columns = ['Date', 'Price']
    price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
    price_df.to_csv("bitcoin_historical_price_data.csv", index=False)
    print(f"Downloaded {len(price_df)} days of Bitcoin price data")

except Exception as e:
    print(f"Error downloading price data: {e}")
    print("Creating synthetic price data for demonstration purposes...")

    # Create synthetic Bitcoin price data if the download fails
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate plausible Bitcoin prices
    base_price = 10000
    volatility = 0.05
    trend = 0.001

    prices = [base_price]
    for i in range(1, len(date_range)):
        # Random walk with drift
        change = prices[-1] * (1 + np.random.normal(trend, volatility))
        prices.append(change)

    # Create DataFrame
    price_df = pd.DataFrame({
        'Date': date_range.date,
        'Price': prices
    })
    price_df.to_csv("bitcoin_historical_price_data.csv", index=False)
    print(f"Created synthetic price data with {len(price_df)} days")

# Part 4: Feature Engineering
# --------------------------
print("Performing feature engineering...")

# Calculate tweet sentiment based on text
print("Running sentiment analysis on tweets...")

# For this demonstration, we'll use a simple lexicon-based approach
# first, then build a deep learning model
positive_words = set([
    'bullish', 'gain', 'profit', 'win', 'moon', 'hodl', 'buy', 'positive',
    'rise', 'up', 'good', 'great', 'excellent', 'increase', 'growth', 'rally',
    'surge', 'climb', 'jump', 'soar', 'peak', 'high', 'record', 'best'
])

negative_words = set([
    'bearish', 'loss', 'crash', 'sell', 'drop', 'down', 'fall', 'negative',
    'decrease', 'decline', 'bear', 'bad', 'worse', 'worst', 'dump', 'plunge',
    'collapse', 'correction', 'panic', 'fear', 'fail', 'poor', 'low', 'terrible'
])

def simple_sentiment(text):
    """Calculate a simple sentiment score based on word matching"""
    if not isinstance(text, str):
        return 0

    words = text.split()
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)

    if pos_count > neg_count:
        return 1  # Positive
    elif neg_count > pos_count:
        return -1  # Negative
    else:
        return 0  # Neutral

# Apply simple sentiment analysis
bitcoin_tweets_df['simple_sentiment'] = bitcoin_tweets_df['cleaned_text'].apply(simple_sentiment)

# First, identify which columns are text vs numeric
numeric_cols = bitcoin_tweets_df.select_dtypes(include=['number']).columns.tolist()
print("Numeric columns:", numeric_cols)

# First, check which columns actually exist in your dataframe
print("Columns in the dataset:", bitcoin_tweets_df.columns.tolist())

# Then check data types
print("Column data types:")
print(bitcoin_tweets_df.dtypes)

# Identify which columns are numeric
numeric_cols = bitcoin_tweets_df.select_dtypes(include=['number']).columns.tolist()
print("Numeric columns:", numeric_cols)

# Check if specific columns exist before using them
has_user_followers = 'user_followers' in bitcoin_tweets_df.columns
has_user_verified = 'user_verified' in bitcoin_tweets_df.columns
has_is_retweet = 'is_retweet' in bitcoin_tweets_df.columns

# Build aggregation dictionary based on what's available
agg_dict = {
    'cleaned_text': lambda x: ' '.join(x),  # Join all tweets for the day
    'text': 'count',                        # Count tweets per day
}

# Only add numeric aggregations for columns that exist and are numeric
if has_user_followers and 'user_followers' in numeric_cols:
    agg_dict['user_followers'] = 'mean'

if has_user_verified and 'user_verified' in numeric_cols:
    agg_dict['user_verified'] = 'sum'

if has_is_retweet and 'is_retweet' in numeric_cols:
    agg_dict['is_retweet'] = 'mean'

# Add simple_sentiment if it exists and is numeric
if 'simple_sentiment' in bitcoin_tweets_df.columns and 'simple_sentiment' in numeric_cols:
    agg_dict['simple_sentiment'] = 'mean'

# Now do the groupby with only compatible columns
daily_tweets = bitcoin_tweets_df.groupby('date_only').agg(agg_dict).reset_index()

# Rename columns
daily_tweets.rename(columns={'text': 'tweet_count', 'date_only': 'Date'}, inplace=True)

daily_tweets.rename(columns={'date_only': 'Date', 'text': 'tweet_count'}, inplace=True)

# Part 5: Merge Tweet Data with Price Data
# --------------------------------------
print("Merging tweet data with price data...")

# Merge tweets with price data
merged_df = pd.merge(daily_tweets, price_df, on='Date', how='inner')
merged_df = merged_df.sort_values('Date')  # Ensure chronological order
merged_df.to_csv("merged_bitcoin_data.csv", index=False)

print(f"Final dataset contains {len(merged_df)} days with both tweets and price data")

# Part 6: Prepare Data for Transformer Model
# --------------------------------------
print("Preparing data for transformer model...")

# Function to encode texts using transformer tokenizer
def encode_texts(texts, max_length=max_len):
    return tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )

# Encode the merged text data
encoded_texts = encode_texts(merged_df['cleaned_text'])
input_ids = encoded_texts['input_ids']
attention_mask = encoded_texts['attention_mask']

# Create target variable: 1 for positive sentiment, 0 for neutral, -1 for negative
# We'll use the price change to determine the "true" sentiment for supervised learning
merged_df['price_shift'] = merged_df['Price'].shift(-1)
merged_df['price_change'] = merged_df['price_shift'] - merged_df['Price']
merged_df['price_movement'] = np.where(merged_df['price_change'] > 0, 1,
                              np.where(merged_df['price_change'] < 0, -1, 0))

# Drop last row since we don't have next day's price
merged_df = merged_df.dropna(subset=['price_change'])

# Prepare features and target
X_input_ids = input_ids[:len(merged_df)]
X_attention_mask = attention_mask[:len(merged_df)]
y_numeric = merged_df['price_movement'].values

# Convert to one-hot encoding for multi-class classification
y = tf.keras.utils.to_categorical(y_numeric + 1)  # Add 1 to map -1,0,1 to 0,1,2

# Split data into training and testing sets
train_size = int(0.8 * len(X_input_ids))
X_train_ids = X_input_ids[:train_size]
X_train_mask = X_attention_mask[:train_size]
y_train = y[:train_size]

X_test_ids = X_input_ids[train_size:]
X_test_mask = X_attention_mask[train_size:]
y_test = y[train_size:]

print(f"Training set size: {train_size}, Test set size: {len(X_input_ids) - train_size}")

# Store the original text for token analysis
X_train_text = merged_df['cleaned_text'][:train_size].tolist()
X_test_text = merged_df['cleaned_text'][train_size:].tolist()

# Part 7: Build and Train Transformer Model - COMPLETELY REBUILT APPROACH
# ------------------------------------------------------
print("Building a compatible transformer-based model...")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Initialize the pre-trained transformer model
transformer = TFAutoModel.from_pretrained(transformer_model_name)

# Create a completely separate model that doesn't try to include the transformer directly in Keras
class SentimentClassifier(tf.keras.Model):
    def __init__(self, num_classes=3):
        super(SentimentClassifier, self).__init__()
        # Define classification layers
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        # Apply classification layers
        x = self.dropout1(inputs, training=training)
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        return self.classifier(x)

# Function to extract BERT features outside of Keras model
def extract_features(input_ids, attention_mask):
    """Extract features from transformer model as TensorFlow tensors"""
    # Use the transformer model directly with TensorFlow tensors
    outputs = transformer(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        training=False  # We don't train the transformer in this step
    )

    # Get the hidden states or pooled output
    if hasattr(outputs, 'last_hidden_state'):
        sequence_output = outputs.last_hidden_state
    else:
        sequence_output = outputs[0]

    # Apply pooling
    # Create a mask to handle padding (convert attention_mask to float)
    mask = tf.cast(attention_mask, tf.float32)
    mask = tf.expand_dims(mask, -1)  # [batch_size, seq_len, 1]

    # Multiply by mask to zero out padding tokens
    masked_output = sequence_output * mask

    # Sum over sequence dimension and divide by non-zero count
    sum_embeddings = tf.reduce_sum(masked_output, axis=1)
    count_tokens = tf.reduce_sum(mask, axis=1)

    # Average the embeddings
    pooled_output = sum_embeddings / tf.maximum(count_tokens, 1.0)

    return pooled_output

# Create the classifier model
classifier = SentimentClassifier(num_classes=3)

# Extract features from training data and convert to NumPy arrays
print("Extracting features from training data...")
train_features_list = []
batch_size = 32

# Process in batches to avoid memory issues
for i in range(0, len(X_train_ids), batch_size):
    end_idx = min(i + batch_size, len(X_train_ids))
    batch_ids = X_train_ids[i:end_idx]
    batch_mask = X_train_mask[i:end_idx]

    # Extract features
    batch_features = extract_features(batch_ids, batch_mask)
    train_features_list.append(batch_features.numpy())

# Concatenate all batches
train_features = np.vstack(train_features_list)

# Extract features from test data
print("Extracting features from test data...")
test_features_list = []

for i in range(0, len(X_test_ids), batch_size):
    end_idx = min(i + batch_size, len(X_test_ids))
    batch_ids = X_test_ids[i:end_idx]
    batch_mask = X_test_mask[i:end_idx]

    # Extract features
    batch_features = extract_features(batch_ids, batch_mask)
    test_features_list.append(batch_features.numpy())

# Concatenate all batches
test_features = np.vstack(test_features_list)

# Compile the classifier
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
classifier.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define early stopping and learning rate reduction
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

# Train the classifier on the extracted features
print("Training the classifier...")
history = classifier.fit(
    train_features,
    y_train,
    epochs=20,
    batch_size=64,  # Can use larger batch size since features are precomputed
    validation_data=(test_features, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Define a complete model for prediction (combining feature extraction and classification)
class CompleteSentimentModel:
    def __init__(self, transformer_model, classifier_model):
        self.transformer = transformer_model
        self.classifier = classifier_model

    def predict(self, inputs):
        """Make predictions using both models"""
        input_ids, attention_mask = inputs

        # Extract features in batches
        features_list = []
        batch_size = 32

        for i in range(0, len(input_ids), batch_size):
            end_idx = min(i + batch_size, len(input_ids))
            batch_ids = input_ids[i:end_idx]
            batch_mask = attention_mask[i:end_idx]

            # Extract features
            batch_features = extract_features(batch_ids, batch_mask)
            features_list.append(batch_features.numpy())

        # Concatenate all batches
        all_features = np.vstack(features_list)

        # Classify
        return self.classifier.predict(all_features)

    def save(self, path):
        """Save the classifier model (transformer can be reloaded separately)"""
        self.classifier.save(path)

# Create the combined model
model = CompleteSentimentModel(transformer, classifier)

# Part 8: Evaluate Model
# -------------------
print("Evaluating model performance...")

# Evaluate on test data
test_loss, test_accuracy = classifier.evaluate(test_features, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Create output directory for visualizations
os.makedirs('bitcoin_results', exist_ok=True)

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Check if accuracy is in history
if 'accuracy' in history.history:
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
elif 'val_accuracy' in history.history:
    plt.subplot(1, 2, 2)
    plt.plot(history.history['val_accuracy'])
    plt.title('Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('bitcoin_results/model_performance.png')

# Part 9: Make Predictions
# ---------------------
print("Making predictions and analyzing results...")

# Make predictions on test data using our classifier
y_pred_proba = classifier.predict(test_features)
y_pred = np.argmax(y_pred_proba, axis=1) - 1  # Convert back to -1, 0, 1

# Get ground truth
y_true = np.argmax(y_test, axis=1) - 1

# Calculate confusion matrix and classification report
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# First check how many unique classes you actually have
unique_classes = np.unique(y_true)
print("Unique classes in your data:", unique_classes)
print("Number of unique classes:", len(unique_classes))

# Modify the classification report to match your actual classes
if len(unique_classes) == 2:
    # If you only have 2 classes (maybe missing the neutral class)
    report = classification_report(
        y_true, y_pred,
        target_names=['Negative', 'Positive']  # Adjust these to match your actual class labels
    )
else:
    # If you have all 3 classes as expected
    report = classification_report(
        y_true, y_pred,
        labels=unique_classes,  # Only use classes that actually exist in your data
        target_names=['Negative', 'Neutral', 'Positive'][:len(unique_classes)]
    )

print("Classification Report:")
print(report)

# Save report to file
with open('bitcoin_results/classification_report.txt', 'w') as f:
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)

# Part 10: Predict Sentiment for All Tweets
# ----------------------------------------
print("Predicting sentiment for all tweets...")

# Check the lengths first
print(f"Length of merged_df: {len(merged_df)}")
print(f"Length of input_ids: {len(input_ids)}")

# Extract features for all data
all_features_list = []

# Option 1: If your merged_df is the correct data and input_ids has an extra row
# Trim input_ids to match merged_df length
all_ids = input_ids[:len(merged_df)]
all_mask = attention_mask[:len(merged_df)]

# Process in batches to avoid memory issues
for i in range(0, len(all_ids), batch_size):
    end_idx = min(i + batch_size, len(all_ids))
    batch_ids = all_ids[i:end_idx]
    batch_mask = all_mask[i:end_idx]

    # Extract features
    batch_features = extract_features(batch_ids, batch_mask)
    all_features_list.append(batch_features.numpy())

# Concatenate all batches
all_features = np.vstack(all_features_list)

# Apply the model to get sentiment
all_predictions = classifier.predict(all_features)
all_sentiment = np.argmax(all_predictions, axis=1) - 1  # Convert back to -1, 0, 1

# Verify lengths match before assigning
print(f"Length of predictions: {len(all_sentiment)}")
print(f"Length of merged_df: {len(merged_df)}")

# Add predictions to the merged dataframe - ONLY if lengths match
if len(all_sentiment) == len(merged_df):
    sentiment_df = merged_df.copy()
    sentiment_df['predicted_sentiment'] = all_sentiment

    # Save the sentiment predictions
    sentiment_df.to_csv('bitcoin_results/bitcoin_sentiment_predictions.csv', index=False)
    print("Sentiment predictions saved successfully.")
else:
    print("ERROR: Length mismatch! Cannot add predictions to dataframe.")
    # Alternative approach: trim to the shorter length
    min_length = min(len(all_sentiment), len(merged_df))
    print(f"Using the first {min_length} entries from both datasets")

    sentiment_df = merged_df.iloc[:min_length].copy()
    sentiment_df['predicted_sentiment'] = all_sentiment[:min_length]

    # Save the sentiment predictions
    sentiment_df.to_csv('bitcoin_results/bitcoin_sentiment_predictions.csv', index=False)
    print(f"Sentiment predictions saved with {min_length} rows (truncated).")

# Part 11: Visualize Results
# -----------------------
print("Creating visualizations...")

# Plot price vs. predicted sentiment
plt.figure(figsize=(16, 8))

# Create twin axes
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot price
ax1.plot(sentiment_df['Date'], sentiment_df['Price'], 'b-', label='Bitcoin Price')

# Calculate 7-day moving average of sentiment
sentiment_df['sentiment_ma7'] = sentiment_df['predicted_sentiment'].rolling(window=7).mean()

# Plot sentiment moving average
ax2.plot(sentiment_df['Date'], sentiment_df['sentiment_ma7'], 'r-', label='7-Day Sentiment MA')

# Plot tweet volume as bubble size
sizes = sentiment_df['tweet_count'] / sentiment_df['tweet_count'].max() * 100
colors = sentiment_df['predicted_sentiment'].apply(lambda x: 'green' if x > 0 else ('red' if x < 0 else 'gray'))
ax2.scatter(sentiment_df['Date'], sentiment_df['predicted_sentiment'],
           s=sizes, alpha=0.4, c=colors, label='Daily Sentiment')

# Customize plot
ax1.set_xlabel('Date')
ax1.set_ylabel('Price (USD)', color='b')
ax2.set_ylabel('Sentiment', color='r')
ax1.grid(True, alpha=0.3)

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Bitcoin Price vs. Tweet Sentiment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('bitcoin_results/bitcoin_price_vs_sentiment.png')

# Part 12: Token Importance Analysis for Transformers
# ---------------------------------------------------
print("Analyzing important tokens for sentiment prediction...")

# For transformers, we need a different approach to understand important tokens
# We'll use a simplified token frequency analysis by sentiment class

# Function to analyze token importance
def analyze_token_importance(texts, sentiments, top_n=20):
    """Analyze which tokens appear most frequently in each sentiment class"""
    from collections import Counter

    # Group texts by sentiment
    pos_texts = [text for text, sent in zip(texts, sentiments) if sent > 0]
    neg_texts = [text for text, sent in zip(texts, sentiments) if sent < 0]
    neu_texts = [text for text, sent in zip(texts, sentiments) if sent == 0]

    # Get words by sentiment
    pos_words = []
    for text in pos_texts:
        pos_words.extend(text.split())

    neg_words = []
    for text in neg_texts:
        neg_words.extend(text.split())

    neu_words = []
    for text in neu_texts:
        neu_words.extend(text.split())

    # Count word frequencies
    pos_counts = Counter(pos_words).most_common(top_n)
    neg_counts = Counter(neg_words).most_common(top_n)
    neu_counts = Counter(neu_words).most_common(top_n)

    return pos_counts, neg_counts, neu_counts

# Analyze token importance for test predictions
print("Analyzing token importance in test predictions...")
pos_counts, neg_counts, neu_counts = analyze_token_importance(
    X_test_text,
    y_pred,
    top_n=50
)

# Save results to CSV
pd.DataFrame(pos_counts, columns=['Token', 'Count']).to_csv(
    'bitcoin_results/positive_tokens.csv', index=False
)
pd.DataFrame(neg_counts, columns=['Token', 'Count']).to_csv(
    'bitcoin_results/negative_tokens.csv', index=False
)
pd.DataFrame(neu_counts, columns=['Token', 'Count']).to_csv(
    'bitcoin_results/neutral_tokens.csv', index=False
)

# Visualize top tokens
plt.figure(figsize=(15, 12))

plt.subplot(3, 1, 1)
plt.barh([token for token, count in pos_counts[:15]][::-1],
         [count for token, count in pos_counts[:15]][::-1],
         color='green')
plt.title('Top 15 Tokens in Positive Sentiment Predictions')
plt.xlabel('Frequency')
plt.ylabel('Token')

plt.subplot(3, 1, 2)
plt.barh([token for token, count in neg_counts[:15]][::-1],
         [count for token, count in neg_counts[:15]][::-1],
         color='red')
plt.title('Top 15 Tokens in Negative Sentiment Predictions')
plt.xlabel('Frequency')
plt.ylabel('Token')

plt.subplot(3, 1, 3)
plt.barh([token for token, count in neu_counts[:15]][::-1],
         [count for token, count in neu_counts[:15]][::-1],
         color='gray')
plt.title('Top 15 Tokens in Neutral Sentiment Predictions')
plt.xlabel('Frequency')
plt.ylabel('Token')

plt.tight_layout()
plt.savefig('bitcoin_results/sentiment_tokens.png')

# Part 13: Save the Model
# ----------------------
print("Saving the model...")

# Save the classifier model
try:
    classifier.save('bitcoin_results/bitcoin_classifier_model')
    print("Classifier model saved successfully")
except Exception as e:
    print(f"Error saving classifier model: {e}")
    print("Saving just the weights...")
    classifier.save_weights('bitcoin_results/bitcoin_classifier.weights.h5')

# Save tokenizer
tokenizer.save_pretrained('bitcoin_results/bitcoin_tokenizer')
print("Tokenizer saved successfully")

# Part 14: Analyze Sentiment Correlation with Price
# ---------------------------------------------
print("Analyzing correlation between sentiment and price movements...")

# Calculate correlations
price_corr = sentiment_df['predicted_sentiment'].corr(sentiment_df['Price'])
price_change_corr = sentiment_df['predicted_sentiment'].corr(sentiment_df['price_change'])

# Calculate lagged correlations
# Does sentiment predict next day's price?
sentiment_df['next_day_price_change'] = sentiment_df['price_change'].shift(-1)
sentiment_to_next_price_corr = sentiment_df['predicted_sentiment'].corr(
    sentiment_df['next_day_price_change']
)

# Does price change predict next day's sentiment?
sentiment_df['next_day_sentiment'] = sentiment_df['predicted_sentiment'].shift(-1)
price_to_next_sentiment_corr = sentiment_df['price_change'].corr(
    sentiment_df['next_day_sentiment']
)

print(f"Correlation between sentiment and price: {price_corr:.4f}")
print(f"Correlation between sentiment and price change: {price_change_corr:.4f}")
print(f"Correlation between sentiment and next day's price change: {sentiment_to_next_price_corr:.4f}")
print(f"Correlation between price change and next day's sentiment: {price_to_next_sentiment_corr:.4f}")

# Save correlation results
with open('bitcoin_results/correlation_analysis.txt', 'w') as f:
    f.write(f"Correlation between sentiment and price: {price_corr:.4f}\n")
    f.write(f"Correlation between sentiment and price change: {price_change_corr:.4f}\n")
    f.write(f"Correlation between sentiment and next day's price change: {sentiment_to_next_price_corr:.4f}\n")
    f.write(f"Correlation between price change and next day's sentiment: {price_to_next_sentiment_corr:.4f}\n")

# Part 15: Analyze Bitcoin Sentiment by Influence
# -------------------------------------------
print("Analyzing sentiment by user influence...")

# First, convert user_followers to numeric
# Use errors='coerce' to convert non-numeric values to NaN
bitcoin_tweets_df['user_followers_numeric'] = pd.to_numeric(bitcoin_tweets_df['user_followers'], errors='coerce')

# Print sample to verify conversion
print("Sample of converted follower counts:")
print(bitcoin_tweets_df[['user_followers', 'user_followers_numeric']].head(10))

# Replace NaN values with 0 or some other appropriate value
bitcoin_tweets_df['user_followers_numeric'] = bitcoin_tweets_df['user_followers_numeric'].fillna(0)

# Group tweets by influence (follower count)
try:
    bitcoin_tweets_df['follower_bucket'] = pd.qcut(
        bitcoin_tweets_df['user_followers_numeric'].clip(upper=1000000),
        q=5,
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
except Exception as e:
    print(f"Error creating follower buckets: {e}")
    # Alternative approach: create buckets manually
    bins = [0, 100, 1000, 10000, 100000, float('inf')]
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    bitcoin_tweets_df['follower_bucket'] = pd.cut(
        bitcoin_tweets_df['user_followers_numeric'],
        bins=bins,
        labels=labels
    )

# Tokenize individual tweets for prediction
print("Tokenizing individual tweets for sentiment prediction...")

# Process in batches to avoid memory issues
batch_size = 1000
all_tweet_sentiments = []

# Get total number of tweets for progress tracking
total_tweets = len(bitcoin_tweets_df)
processed_tweets = 0

# Process in batches
for i in range(0, total_tweets, batch_size):
    # Get batch of tweets
    batch_tweets = bitcoin_tweets_df['cleaned_text'][i:i+batch_size].tolist()

    # Tokenize batch
    batch_encoded = tokenizer(
        batch_tweets,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='tf'
    )

    # Extract features
    batch_features_list = []

    # Process in smaller sub-batches if needed
    sub_batch_size = 32
    for j in range(0, len(batch_tweets), sub_batch_size):
        end_j = min(j + sub_batch_size, len(batch_tweets))
        sub_ids = batch_encoded['input_ids'][j:end_j]
        sub_mask = batch_encoded['attention_mask'][j:end_j]

        # Extract features
        sub_features = extract_features(sub_ids, sub_mask)
        batch_features_list.append(sub_features.numpy())

    # Concatenate all sub-batches
    batch_features = np.vstack(batch_features_list)

    # Predict sentiment
    batch_preds = classifier.predict(batch_features)
    batch_sentiment = np.argmax(batch_preds, axis=1) - 1  # Convert back to -1, 0, 1

    # Add to results
    all_tweet_sentiments.extend(batch_sentiment)

    # Update progress
    processed_tweets += len(batch_tweets)
    print(f"Processed {processed_tweets}/{total_tweets} tweets...")

# Add predicted sentiment to DataFrame
bitcoin_tweets_df['predicted_sentiment'] = all_tweet_sentiments[:len(bitcoin_tweets_df)]

# Analyze sentiment by influence
influence_sentiment = bitcoin_tweets_df.groupby('follower_bucket')['predicted_sentiment'].agg(['mean', 'count'])
influence_sentiment.columns = ['Average Sentiment', 'Number of Tweets']
influence_sentiment = influence_sentiment.reset_index()

# Plot sentiment by influence
plt.figure(figsize=(10, 6))
bars = plt.bar(influence_sentiment['follower_bucket'], influence_sentiment['Average Sentiment'])

# Color bars by sentiment
for i, bar in enumerate(bars):
    if influence_sentiment['Average Sentiment'][i] > 0:
        bar.set_color('green')
    elif influence_sentiment['Average Sentiment'][i] < 0:
        bar.set_color('red')
    else:
        bar.set_color('gray')

plt.title('Bitcoin Tweet Sentiment by User Influence (Follower Count)')
plt.xlabel('User Influence Level')
plt.ylabel('Average Sentiment')
plt.grid(axis='y', alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('bitcoin_results/sentiment_by_influence.png')

# Part 16: Create a Sentiment Heatmap by Date
# -----------------------------------------
print("Creating temporal sentiment heatmap...")

# Group by year-month for temporal analysis
bitcoin_tweets_df['year_month'] = bitcoin_tweets_df['date'].dt.to_period('M')
monthly_sentiment = bitcoin_tweets_df.groupby('year_month')['predicted_sentiment'].mean().reset_index()
monthly_sentiment['year_month_str'] = monthly_sentiment['year_month'].astype(str)

# Plot monthly sentiment over time
plt.figure(figsize=(14, 6))
bars = plt.bar(monthly_sentiment['year_month_str'], monthly_sentiment['predicted_sentiment'])

# Color bars by sentiment
for i, bar in enumerate(bars):
    if monthly_sentiment['predicted_sentiment'][i] > 0:
        bar.set_color('green')
    elif monthly_sentiment['predicted_sentiment'][i] < 0:
        bar.set_color('red')
    else:
        bar.set_color('gray')

plt.title('Bitcoin Tweet Sentiment by Month')
plt.xlabel('Month')
plt.ylabel('Average Sentiment')
plt.grid(axis='y', alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('bitcoin_results/sentiment_by_month.png')

# Part 17: Analyze Transformer Embedding Space
# ------------------------------------------
print("Analyzing transformer embedding space...")

try:
    # Sample tweets and get embeddings
    sample_size = min(500, len(X_test_text))  # Limit for visualization
    sample_indices = np.random.choice(len(X_test_text), sample_size, replace=False)

    sample_texts = [X_test_text[i] for i in sample_indices]
    sample_sentiments = y_pred[sample_indices]

    # Tokenize the sample
    sample_encoded = tokenizer(
        sample_texts,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='tf'
    )

    # Get embeddings from the extracted features
    sample_features_list = []

    # Process in batches
    for i in range(0, len(sample_texts), batch_size):
        end_idx = min(i + batch_size, len(sample_texts))
        batch_ids = sample_encoded['input_ids'][i:end_idx]
        batch_mask = sample_encoded['attention_mask'][i:end_idx]

        # Extract features
        batch_features = extract_features(batch_ids, batch_mask)
        sample_features_list.append(batch_features.numpy())

    # Concatenate all batches
    embeddings = np.vstack(sample_features_list)

    # Reduce dimensionality for visualization
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # First reduce with PCA
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)

    # Then apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_pca)

    # Create visualization
    plt.figure(figsize=(12, 10))

    # Define sentiment colors
    colors = ['red' if s < 0 else ('gray' if s == 0 else 'green') for s in sample_sentiments]

    # Plot points
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.7)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Positive'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Neutral'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Negative')
    ]
    plt.legend(handles=legend_elements)

    plt.title('Bitcoin Tweet Embeddings (t-SNE Visualization)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.savefig('bitcoin_results/embedding_visualization.png')

    print("Embedding visualization saved to bitcoin_results/embedding_visualization.png")

except Exception as e:
    print(f"Embedding analysis failed: {e}")
    print("Skipping embedding visualization...")

print("\nAnalysis complete! Results saved to the 'bitcoin_results' directory.")
print("\nSummary of files generated:")
print("- bitcoin_historical_price_data.csv: Bitcoin price data")
print("- merged_bitcoin_data.csv: Combined tweets and price data")
print("- bitcoin_results/model_performance.png: Training metrics")
print("- bitcoin_results/bitcoin_price_vs_sentiment.png: Price and sentiment visualization")
print("- bitcoin_results/sentiment_tokens.png: Top tokens by sentiment class")
print("- bitcoin_results/bitcoin_classifier_model: Saved classifier model")
print("- bitcoin_results/bitcoin_classifier_weights.h5: Model weights (backup)")
print("- bitcoin_results/bitcoin_tokenizer: Saved tokenizer")
print("- bitcoin_results/bitcoin_sentiment_predictions.csv: Daily sentiment predictions")
print("- bitcoin_results/sentiment_by_influence.png: Sentiment by user influence")
print("- bitcoin_results/sentiment_by_month.png: Monthly sentiment heatmap")
print("- bitcoin_results/positive_tokens.csv: Important tokens for positive sentiment")
print("- bitcoin_results/negative_tokens.csv: Important tokens for negative sentiment")
print("- bitcoin_results/neutral_tokens.csv: Important tokens for neutral sentiment")
print("- bitcoin_results/correlation_analysis.txt: Price-sentiment correlation metrics")
print("- bitcoin_results/embedding_visualization.png: Transformer embedding space visualization")