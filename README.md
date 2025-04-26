# Project Structure and File Descriptions

## data
This folder mainly contains three files:
- **btcprice**: Used to obtain Bitcoin price data.
- **reddit_BTC**: Collects Reddit posts whose titles contain Bitcoin-related keywords.
- **reddit_cmt_btc**: Extracts comments from the collected Reddit posts.

## analysis
This folder compares the dictionary-based sentiment method and the Neural Bag of Words (NBoW) method.

### Dictionary Method
- **dictionary_sentiment_analysis.py**: Performs sentiment analysis using a dictionary method and calculates a weighted average sentiment for each day based on upvotes.
- **price_and DICsentiment.py**: After obtaining the sentiment output, combines price and daily weighted average sentiment, and uses Random Forest (classification) to predict future price movements. Time lag is considered to find the optimal lag for prediction.

### Neural Bag of Words (NBoW) Method
- **NBoW.py**: Trains the NBoW model using crypto news datasets from Kaggle, achieving high training accuracy. The trained model is saved as `nbow.pt`.
- **predict_sentiment.py**: Uses the trained `nbow.pt` model to predict sentiment in Reddit comment datasets.
- **price_and_NBOWsentiment.py**: After obtaining the sentiment output, combines price and daily weighted average sentiment, and uses Random Forest (classification) to predict future price movements. Time lag is also considered to find the optimal lag for prediction.

---

**Note:**
- All scripts are designed for reproducibility and modular analysis.
- The project supports both dictionary-based and neural network-based sentiment analysis pipelines for cryptocurrency-related social media data.
