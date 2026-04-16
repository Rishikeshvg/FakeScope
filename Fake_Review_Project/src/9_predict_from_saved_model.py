import joblib

# Load saved models and vectorizer
logistic_model = joblib.load("logistic_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
rf_model = joblib.load("random_forest_model.pkl")


def extract_stylometric_features(text):
    word_count = len(text.split())
    excl_count = text.count('!')
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    return [[word_count, excl_count, caps_ratio]]


def predict_review(text):
    # --- TF-IDF Prediction ---
    text_vec = tfidf.transform([text])
    tfidf_pred = logistic_model.predict(text_vec)[0]
    tfidf_prob = logistic_model.predict_proba(text_vec)[0][tfidf_pred]

    # --- Stylometric Prediction ---
    style_features = extract_stylometric_features(text)
    rf_pred = rf_model.predict(style_features)[0]
    rf_prob = rf_model.predict_proba(style_features)[0][rf_pred]

    # --- Combine Decisions (Simple Majority Logic) ---
    if tfidf_pred == 1 or rf_pred == 1:
        final_label = "Deceptive"
    else:
        final_label = "Truthful"

    avg_confidence = (tfidf_prob + rf_prob) / 2

    return final_label, avg_confidence, tfidf_pred, rf_pred


if __name__ == "__main__":
    user_input = input("Enter a review: ")

    final_label, confidence, tfidf_pred, rf_pred = predict_review(user_input)

    print("\nTF-IDF Model Prediction:", "Deceptive" if tfidf_pred == 1 else "Truthful")
    print("Random Forest Prediction:", "Deceptive" if rf_pred == 1 else "Truthful")
    print("\nFinal Combined Prediction:", final_label)
    print("Confidence:", round(confidence * 100, 2), "%")
