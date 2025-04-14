import pandas as pd
import statsmodels.api as sm

# Step 1: Load the Excel file with merged data
# Replace 'daily_negativity_and_prices.xlsx' with your actual file path if different
input_file = 'daily_negativity_and_prices.xlsx'
df = pd.read_excel(input_file)

# Step 2: Set timeframe (replace with desired start and end dates)
start_date = pd.to_datetime('2022-01-01').date()  # Example start date
end_date = pd.to_datetime('2022-05-15').date()    # Example end date | important: cutoff after crash

# Filter data based on timeframe
df = df[(pd.to_datetime(df['Comment Date']).dt.date >= start_date) & 
        (pd.to_datetime(df['Comment Date']).dt.date <= end_date)]

# Step 3: Prepare the data for regression
# Independent variable(s): avg_negativity_score
# Dependent variable: Price
X = df[['avg_negativity_score']]  # Independent variable
X = sm.add_constant(X)  # Add a constant term for the intercept
y = df['Price']  # Dependent variable

# Step 4: Run multiple regression
model = sm.OLS(y, X).fit()

# Step 5: Display regression results
print("\nMultiple Regression Results:")
print(model.summary())

# Step 6: Save the results to a new Excel file with Comment Date included
results_df = pd.DataFrame({
    'Comment Date': df['Comment Date'],
    'avg_negativity_score': df['avg_negativity_score'],
    'Price': df['Price'],
    'Predicted_Price': model.predict(X)
})
results_df.to_excel('regression_results.xlsx', index=False)
print("\nRegression results saved to 'regression_results.xlsx'")

