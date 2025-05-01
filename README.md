# Project Structure and File Descriptions
## BTC DATA file
result and outcomes
## src file
### data
This folder mainly contains three files:
- **btcprice**: Used to obtain Bitcoin price data.
- **reddit_BTC**: Collects Reddit posts whose titles contain Bitcoin-related keywords.
- **reddit_cmt_btc**: Extracts comments from the collected Reddit posts.

### analysis
This folder compares the dictionary-based sentiment method and the Neural Bag of Words (NBoW) method.

#### Dictionary Method
- **dictionary_sentiment_analysis.py**: Performs sentiment analysis using a dictionary method and calculates a weighted average sentiment for each day based on upvotes.
- **price_and DICsentiment.py**: After obtaining the sentiment output, combines price and daily weighted average sentiment, and uses Random Forest (classification) to predict future price movements. Time lag is considered to find the optimal lag for prediction.

#### Neural Bag of Words (NBoW) Method
- **NBoW.py**: Trains the NBoW model using crypto news datasets from Kaggle, achieving high training accuracy. The trained model is saved as `nbow.pt`.
- **predict_sentiment.py**: Uses the trained `nbow.pt` model to predict sentiment in Reddit comment datasets.
- **price_and_NBOWsentiment.py**: After obtaining the sentiment output, combines price and daily weighted average sentiment, and uses Random Forest (classification) to predict future price movements. Time lag is also considered to find the optimal lag for prediction.

---

**Note:**
- All scripts are designed for reproducibility and modular analysis.
- The project supports both dictionary-based and neural network-based sentiment analysis pipelines for cryptocurrency-related social media data.

# File Format Conventions

- No Excel files allowed, only csv --> much better handling

## Price Files

Date,Time,Timezone,Price,Volume,Market_cap

## Sentiment Files

### Reddit Posts

date,title,post_url,subreddit,post_id

### Reddit Comments

date,comment,subreddit,comment_id,upvotes,post_url

## Sentiment X Time Files 

## Sentiment X Price Files

# Naming Conventions

## Sentiment Files 

tokeninsightid_posts -> post on reddit that refer to the respective crytocurrency
tokeninsightid_comments -> comments below that posts, also filtered for respective cryptocurrency
TICKER_sentiment_training -> training datasets obtained from kaggle to test accuracy of 

## Price Files

tokeninsightid_price

## Sentiment X Time

TICKER_sentiment_x_time

## Sentiment X Price

DICT_TICKER_sentiment_x_price

# Workflow 

## Price data

- Pricedata is collected and constantly updated with tokeninsight API

## Alpha Vantage Data

- News headlines referring to the respective cryptocurrency are collected from Alpha Vantage

## Reddit Data

- Comments are retrieved from the respective cryptocurrencies forum
- Data Cleaning is done on comments to cut out noise comments that have nothing to do with the actual cryptocurrency discussed in the forum

# Pending Improvements

- currently, sentiment data collection is connected to names of price files (=spaghetti code) 
--> actually, there should be one file only to store the available cryptocurrencies, and this file should be referred to. That would be more logical. 
- there should maybe be one central module that contains all the file paths, API keys, and other important central variables, and in other modules the file paths etc. should be referred by using the variables in that module --> then it is easier to restructure the code an make changes
