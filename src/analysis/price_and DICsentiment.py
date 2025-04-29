import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data
data_path = r"D:\sjtucyx iii\2425\FINA4350\BTC\BTC DATA\price_sentiment.xlsx"
data = pd.read_excel(data_path)

# Data preprocessing
# Assuming 'Close' is closing price and 'Weighted_Sentiment' is sentiment score
# Create new column 'Price_Change' indicating price increase (1) or decrease (0)
data['Price_Change'] = (data['Close'].diff() > 0).astype(int)

results = []

for lag_days in range(16):
    # lag all features
    data['lag_Weighted_Sentiment'] = data['Weighted_Sentiment'].shift(lag_days)
    data['lag_Simple_Avg_Sentiment'] = data['Simple_Avg_Sentiment'].shift(lag_days)
    data['lag_Vol'] = data['Vol'].shift(lag_days)
    # remove missing
    temp = data.dropna(subset=['lag_Weighted_Sentiment', 'lag_Simple_Avg_Sentiment', 'lag_Vol', 'Price_Change'])
    features = temp[['lag_Weighted_Sentiment', 'lag_Simple_Avg_Sentiment', 'lag_Vol']]
    labels = temp['Price_Change']
    # time series not shuffle
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=False)
    # random forest
    rf_model = RandomForestClassifier(n_estimators=300, random_state=1029)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((lag_days, acc))
    print(f"lag_days={lag_days}, Accuracy={acc:.4f}")

# best lag days and corresponding accuracy
best_lag, best_acc = max(results, key=lambda x: x[1])
print(f"\nBest lag days: {best_lag}, Corresponding accuracy: {best_acc:.4f}")

# output the feature importance of the best lag days
data['lag_Weighted_Sentiment'] = data['Weighted_Sentiment'].shift(best_lag)
data['lag_Simple_Avg_Sentiment'] = data['Simple_Avg_Sentiment'].shift(best_lag)
data['lag_Vol'] = data['Vol'].shift(best_lag)
temp = data.dropna(subset=['lag_Weighted_Sentiment', 'lag_Simple_Avg_Sentiment', 'lag_Vol', 'Price_Change'])
features = temp[['lag_Weighted_Sentiment', 'lag_Simple_Avg_Sentiment', 'lag_Vol']]
labels = temp['Price_Change']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=False)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
feature_importances = rf_model.feature_importances_
for feature, importance in zip(features.columns, feature_importances):
    print(f"{feature}: {importance:.4f}")

# Save all lag results to a txt file
with open('BTC/BTC DATA/output/price_and_DICsentiment_results.txt', 'w') as f:
    f.write(f"Best lag days: {best_lag}, Corresponding accuracy: {best_acc:.4f}\n")
    f.write("Feature Importances:\n")
    for feature, importance in zip(features.columns, feature_importances):
        f.write(f"{feature}: {importance:.4f}\n")



