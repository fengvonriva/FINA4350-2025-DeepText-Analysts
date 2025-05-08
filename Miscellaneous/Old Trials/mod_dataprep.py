import pandas as pd

def filter_reddit_comments(input_path, output_path='filtered_reddit_selected_rows.xlsx', keywords=['terra', 'luna']):
    """
    Filter Reddit comments containing specified keywords (case-insensitive) and save to a new Excel file.

    Parameters:
    - input_path (str): Path to the input Excel file with Reddit comments.
    - output_path (str): Path to save the filtered Excel file (default: 'filtered_reddit_selected_rows.xlsx').
    - keywords (list): List of keywords to filter comments (default: ['terra', 'luna']).

    Returns:
    - pd.DataFrame: Filtered DataFrame containing comments with the specified keywords.
    """
    # Step 1: Load the Excel file with Reddit comments
    comments_df = pd.read_excel(input_path)

    # Step 2: Filter comments containing any of the keywords (case-insensitive)
    pattern = '|'.join(keywords)
    filtered_comments_df = comments_df[comments_df['Comment Text'].str.contains(
        pattern, case=False, na=False)].copy()

    # Step 3: Save to a new Excel file
    filtered_comments_df.to_excel(output_path, index=False)
    print(f"\nResults saved to '{output_path}'")

    return filtered_comments_df

# Placeholder for additional functions (e.g., for Kaggle data)
def process_kaggle_data(input_path, output_path='processed_kaggle_data.csv'):
    """
    Placeholder function for processing Kaggle data.
    Customize this function based on specific Kaggle data processing needs.

    Parameters:
    - input_path (str): Path to the input Kaggle data file.
    - output_path (str): Path to save the processed data (default: 'processed_kaggle_data.csv').

    Returns:
    - pd.DataFrame: Processed Kaggle DataFrame (to be implemented).
    """
    # Placeholder implementation
    print(f"Processing Kaggle data from '{input_path}' (not implemented yet)")
    # Example: df = pd.read_csv(input_path)
    # Process df as needed
    # df.to_csv(output_path, index=False)
    # print(f"\nResults saved to '{output_path}'")
    # return df
    return pd.DataFrame()  # Return empty DataFrame as placeholder

import pandas as pd

def preprocess_price_data(
    input_path='P2_Data_Analysis/Pricedata/LUNA_price.csv',
    output_path='P2_Data_Analysis/Pricedata/LUNA_price_processed.csv'
):
    """
    Process a price data CSV to split the Date column (format: 'YYYY-MM-DD HH:MM:SS UTC+0')
    into Date (YYYY-MM-DD), Time (HH:MM:SS), and Timezone (UTC+0) columns.

    Parameters:
    - input_path (str): Path to the input price data CSV.
    - output_path (str): Path to save the processed CSV.

    Returns:
    - pd.DataFrame: Processed DataFrame with updated columns.
    """
    # Load the CSV
    df = pd.read_csv(input_path)

    # Parse the Date column and split into components
    # Assuming format: '2025-02-26 00:00:00 UTC+0'
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S UTC+0')
    
    # Extract YYYY-MM-DD for Date
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # Extract HH:MM:SS for Time
    df['Time'] = df['Date'].apply(lambda x: pd.to_datetime(x).strftime('%H:%M:%S'))
    
    # Set Timezone to UTC+0
    df['Timezone'] = 'UTC+0'

     # Reorder columns to: Date,Time,Timezone,Price,Volume,Market_cap
    df = df[['Date', 'Time', 'Timezone', 'Price', 'Volume', 'Market_cap']]

    # Save to new CSV
    df.to_csv(output_path, index=False)
    print(f"\nProcessed price data saved to '{output_path}'")

    return df   

if __name__ == "__main__":
    # Example usage for testing
    # filter_reddit_comments('reddit_selected_rows.xlsx')
    preprocess_price_data()