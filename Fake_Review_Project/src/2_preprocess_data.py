import pandas as pd
import re
import os

def clean_text(text):
    """
    Light preprocessing only:
    - Lowercase
    - Remove URLs, HTML tags, extra whitespace
    - Keep punctuation, stopwords, all words intact
    - No lemmatization, no stopword removal
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'<.*?>', '', text)                  # remove HTML tags
    text = re.sub(r'[^a-z\s]', ' ', text)             # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()           # clean whitespace
    return text

# Paths
input_path = os.path.join('data', 'processed', 'combined_dataset.csv')
output_path = os.path.join('data', 'processed', 'combined_cleaned.csv')

try:
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} reviews.")

    df = df.dropna(subset=['text'])
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df[['cleaned_text', 'deceptive']]

    df.to_csv(output_path, index=False)

    print(f"✅ Saved to {output_path}")
    print("\n--- Sample ---")
    print(df['cleaned_text'].iloc[0][:150])

except Exception as e:
    print(f"❌ Error: {e}")