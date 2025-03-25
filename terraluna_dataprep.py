import pandas as pd

# Step 1: Load the Excel file with Reddit comments
# Replace 'reddit_comments.xlsx' with your actual file path
comments_df = pd.read_excel('reddit_selected_rows.xlsx')

# Step 2: Transform data to filter comments containing "terra" or "luna" (case-insensitive)
def filter_terra_luna_comments(df):
    return df[df['Comment Text'].str.contains('terra|luna', case=False, na=False)].copy()

filtered_comments_df = filter_terra_luna_comments(comments_df)

# Step 3: Save to a new Excel file
filtered_comments_df.to_excel('filtered_reddit_selected_rows.xlsx', index=False)
print("\nResults saved to 'filtered_reddit_selected_rows.xlsx'")
