import pandas as pd
import os

# Define the path to your data
data_path = os.path.join('data', 'raw', 'deceptive_data.csv')

try:
    # Load the dataset
    df = pd.read_csv(data_path)
    
    print("✅ Dataset Loaded Successfully!")
    print(f"Total Reviews: {len(df)}")
    print("\n--- Column Names ---")
    print(df.columns.tolist())
    
    print("\n--- First 3 Rows ---")
    print(df[['deceptive', 'text']].head(3))
    
    # Check the balance of Real vs Fake
    print("\n--- Data Distribution ---")
    print(df['deceptive'].value_counts())

except FileNotFoundError:
    print(f"❌ Error: Could not find the file at {data_path}")
    print("Make sure you moved the Kaggle file to data/raw/ and named it deceptive_data.csv")