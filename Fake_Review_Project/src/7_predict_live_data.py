import pandas as pd
import spacy
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Paths
RAW_SCRAPED = os.path.join('data', 'raw', 'scraped_reviews.csv')
CLEANED_TRAIN = os.path.join('data', 'processed', 'cleaned_data.csv')
OUTPUT_PREDICTIONS = os.path.join('data', 'processed', 'live_predictions.csv')

print("Loading NLP tools...")
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(tokens)

try:
    # 1. Train Model
    print("Training on Gold Standard Data...")
    train_df = pd.read_csv(CLEANED_TRAIN).dropna(subset=['cleaned_text'])
    tfidf = TfidfVectorizer(max_features=2000)
    X_train = tfidf.fit_transform(train_df['cleaned_text'])
    y_train = train_df['deceptive'].apply(lambda x: 1 if x == 'deceptive' else 0)
    model = LogisticRegression().fit(X_train, y_train)

    # 2. Load Scraped Data
    live_df = pd.read_csv(RAW_SCRAPED)
    
    # NEW: Better filter to remove "Login", "Cart", and system headers
    garbage_keywords = ['login', 'signup', 'flipkart', 'profile', 'orders', 'notification']
    live_df = live_df[~live_df['text'].str.lower().str.contains('|'.join(garbage_keywords))]
    # Keep only reasonably long sentences
    live_df = live_df[live_df['text'].str.len() > 30]

    print(f"Cleaning {len(live_df)} actual reviews...")
    live_df['cleaned_text'] = live_df['text'].apply(clean_text)
    
    # 3. Predict
    X_live = tfidf.transform(live_df['cleaned_text'])
    live_df['prediction_score'] = model.predict_proba(X_live)[:, 1]
    live_df['is_deceptive'] = live_df['prediction_score'].apply(lambda x: "DECEPTIVE" if x > 0.5 else "TRUTHFUL")

    live_df.to_csv(OUTPUT_PREDICTIONS, index=False)
    
    print("\n" + "="*50)
    print("FINAL CAPSTONE LIVE REPORT")
    print("="*50)
    for i, row in live_df.head(5).iterrows():
        print(f"\nReview: {row['text'][:80]}...")
        print(f"Result: {row['is_deceptive']} (Score: {row['prediction_score']:.2f})")
    
    print("\n" + "="*50)
    print(f"Total Actual Reviews Analyzed: {len(live_df)}")
    print(f"Suspicious Content Found: {len(live_df[live_df['is_deceptive'] == 'DECEPTIVE'])}")

except Exception as e:
    print(f"❌ Error: {e}")