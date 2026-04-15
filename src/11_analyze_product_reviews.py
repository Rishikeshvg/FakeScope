import pandas as pd
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

analyzer = SentimentIntensityAnalyzer()

# Load models
logistic_model = joblib.load("logistic_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
rf_model = joblib.load("random_forest_model.pkl")


def extract_stylometric_features(text):
    word_count = len(text.split())
    excl_count = text.count('!')
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    avg_sentence_length = len(text.split()) / max(1, len(re.split(r'[.!?]', text)))
    first_person_ratio = sum(text.lower().count(p) for p in [" i ", " me ", " my ", " mine "]) / max(1, len(text.split()))
    sentiment_intensity = abs(analyzer.polarity_scores(text)['compound'])

    return [[
        word_count,
        excl_count,
        caps_ratio,
        avg_sentence_length,
        first_person_ratio,
        sentiment_intensity
    ]]


df = pd.read_csv("data/processed/extracted_reviews.csv")

results = []

for review in df["review_text"]:
    text_vec = tfidf.transform([review])
    tfidf_pred = logistic_model.predict(text_vec)[0]

    style_features = extract_stylometric_features(review)
    rf_pred = rf_model.predict(style_features)[0]

    # AND logic for hybrid
    final_label = 1 if (tfidf_pred == 1 and rf_pred == 1) else 0

    results.append(final_label)

total = len(results)
deceptive = sum(results)
fake_percent = (deceptive / total) * 100
trust_score = 100 - fake_percent

print("\n========== PRODUCT ANALYSIS ==========")
print("Total Reviews:", total)
print("Predicted Deceptive:", deceptive)
print("Fake %:", round(fake_percent, 2), "%")
print("Trust Score:", round(trust_score, 2), "/ 100")
