import pandas as pd
import os
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def extract_features(df):
    # Basic features
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['excl_count'] = df['text'].apply(lambda x: x.count('!'))
    df['caps_ratio'] = df['text'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )

    # Average sentence length
    df['avg_sentence_length'] = df['text'].apply(
        lambda x: len(x.split()) / max(1, len(re.split(r'[.!?]', x)))
    )

    # First-person pronoun ratio
    df['first_person_ratio'] = df['text'].apply(
        lambda x: sum(x.lower().count(p) for p in [" i ", " me ", " my ", " mine "]) / max(1, len(x.split()))
    )

    # Sentiment intensity
    df['sentiment_intensity'] = df['text'].apply(
        lambda x: abs(analyzer.polarity_scores(x)['compound'])
    )

    return df


# Paths
input_path = os.path.join('data', 'processed', 'combined_dataset.csv')
output_path = os.path.join('data', 'processed', 'final_features.csv')

try:
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['text'])

    print("Extracting enhanced stylometric features...")
    df = extract_features(df)

    df.to_csv(output_path, index=False)
    print("Feature extraction complete.")
    print("Saved to:", output_path)

except Exception as e:
    print("Error:", e)
