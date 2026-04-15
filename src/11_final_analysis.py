import pandas as pd
import spacy
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Paths
SCRAPED_DATA = os.path.join('data', 'raw', 'scraped_reviews.csv')
TRAIN_DATA = os.path.join('data', 'processed', 'cleaned_data.csv')
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(tokens)

print("🧠 Training Model on Gold Standard Research Data...")
train_df = pd.read_csv(TRAIN_DATA).dropna(subset=['cleaned_text'])
tfidf = TfidfVectorizer(max_features=2000)
X_train = tfidf.fit_transform(train_df['cleaned_text'])
y_train = train_df['deceptive'].apply(lambda x: 1 if x == 'deceptive' else 0)
model = LogisticRegression().fit(X_train, y_train)

print(f"📖 Loading {SCRAPED_DATA}...")
live_df = pd.read_csv(SCRAPED_DATA)
live_df['cleaned'] = live_df['text'].apply(clean_text)

print("🔍 Predicting Authenticity...")
X_live = tfidf.transform(live_df['cleaned'])
live_df['deception_score'] = model.predict_proba(X_live)[:, 1]
live_df['label'] = live_df['deception_score'].apply(lambda x: "🚩 DECEPTIVE" if x > 0.5 else "✅ TRUTHFUL")

# Save results
output_path = os.path.join('data', 'processed', 'final_results.csv')
live_df.to_csv(output_path, index=False)

print("\n" + "="*50)
print("FINAL LIVE DATA REPORT")
print("="*50)
print(live_df[['text', 'status', 'label', 'deception_score']].head(10))