# This file is the central file for all the code execution

# Step 1: Filtering sentiment data specific to currency (e.g. one big reddit dataset, one big kaggl dataset, or easier: tailored datasets) 
# --> mod_dataprep.py
'''Can involve AI-Model (AI has to detect what the comments are about)
Simple way: just controlling case-insensitively for a few words, e.g. Bitcoin, BTC or LUNA, Terraluna, Terra '''

# Step 2: Capturing sentiments and putting it in relation, getting sentiment_x_price data --> mod_sentiment.py
'''
a) Using dictionary method
b) Using AI-Model
'''

# Step 3: Comparing Sentiment with price behaviour --> mod_price
'''
drawing on price data, finding out relationship with price and sentiment, plotting a graph'''

# Step 4: Testing performance of sentiment based trading  --> mod_trading

# Refinement consists of continuously trying to capture the sentiment better --> only then we have a valid result

# The result itself does not matter if the method is valid and we capture Reddit Sentiment accurately - either we can use it to predict price
# or we cannot --> but at least we know then! So it can be great to test one's beliefs
