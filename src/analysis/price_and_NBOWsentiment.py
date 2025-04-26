import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data
data_path = r"D:\sjtucyx iii\2425\FINA4350\BTC\BTC DATA\output\reddit_daily_sentiment.xlsx"
data = pd.read_excel(data_path, sheet_name='Sheet2')

# Data preprocessing
# Create new column 'Price_Change' indicating price increase (1) or decrease (0)
data['Price_Change'] = (data['Price'].diff() > 0).astype(int)

results = []

for lag_days in range(16):
    # lag all features
    data['lagged_sentiment'] = data['weighted_avg_sentiment'].shift(lag_days)
    # remove missing
    temp = data.dropna(subset=['lagged_sentiment', 'Price_Change'])
    features = temp[['lagged_sentiment']]
    labels = temp['Price_Change']
    # time series not shuffle
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=False)
    # Build Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_features='sqrt',
        random_state=1029
    )
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((lag_days, acc))
    print(f"lag_days={lag_days}, Accuracy={acc:.4f}")

# best lag days and corresponding accuracy
best_lag, best_acc = max(results, key=lambda x: x[1])
print(f"\nBest lag days: {best_lag}, Corresponding accuracy: {best_acc:.4f}")

# Feature importance
feature_importances = rf_model.feature_importances_
for feature, importance in zip(features.columns, feature_importances):
    print(f"{feature}: {importance:.4f}")


with open('BTC/BTC DATA/output/price_and_NBOWsentiment_results.txt', 'w') as f:
    f.write(f"Best lag days: {best_lag}, Corresponding accuracy: {best_acc:.4f}\n")
    f.write("Feature Importances:\n")
    for feature, importance in zip(features.columns, feature_importances):
        f.write(f"{feature}: {importance:.4f}\n")

