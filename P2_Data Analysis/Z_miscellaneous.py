# Install required libraries

import pandas as pd

# Step 1: Load the CSV file with explicit delimiter and quoting handling
# Replace 'terra-historical-day-data-all-tokeninsight.csv' with your actual file path
input_file = 'terra-historical-day-data-all-tokeninsight.csv'
df = pd.read_csv(input_file, quotechar='"', delimiter=',', header=0)

# Step 2: Remove all quote characters from the DataFrame
for column in df.columns:
    df[column] = df[column].astype(str).str.replace('"', '')

# Step 3: Remove quotes from column names
df.columns = [col.replace('"', '') for col in df.columns]

# Debug: Print the DataFrame to check structure
print("DataFrame columns:", df.columns.tolist())
print("First few rows:\n", df.head())

# Step 4: Save to an Excel file
output_file = 'terra_cleaned_no_quotes.xlsx'
df.to_excel(output_file, index=False)
print(f"Data saved to {output_file}")



# Old code
'''
# Step 5: Display results
print("\nReddit Comments with Negativity Scores:")
# Show relevant columns: Post Title, Comment Time, Comment Text, and negativity score
print(comments_df[['Post Title', 'Comment Time', 'Comment Text', 'negativity_score']])

# Step 6: Save results to a new CSV file
comments_df.to_csv('reddit_selected_rows_with_scores.csv', index=False)
print("\nResults saved to 'reddit_selected_rows_with_scores.csv'")

# Optional: Save back to Excel if preferred
comments_df.to_excel('reddit_selected_rows_with_scores.xlsx', index=False)
print("Results saved to 'reddit_selected_rows_with_scores.xlsx'")
'''