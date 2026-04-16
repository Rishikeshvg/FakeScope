import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import numpy as np

data_path = os.path.join('data', 'processed', 'combined_cleaned.csv')

try:
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['cleaned_text'])
    print(f"Loaded {len(df)} reviews.")

    X = df['cleaned_text']
    y = df['deceptive']

    # 1. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. TF-IDF fit on training data only
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"Vocabulary size: {len(tfidf.get_feature_names_out())}")

    # 3. Train model
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_tfidf, y_train)

    # 4. Evaluate on test set
    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]

    print("\n--- TF-IDF + Logistic Regression Performance ---")
    print(classification_report(y_test, y_pred, target_names=['Truthful', 'Deceptive']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # 5. Proper cross-validation using Pipeline
    # TF-IDF is refit independently inside each fold - no leakage
    print("\nRunning 5-Fold Cross-Validation (Pipeline - no leakage)...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(max_iter=1000, C=1.0))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    print(f"CV Scores: {cv_scores}")
    print(f"Mean: {cv_scores.mean():.4f} | Std: {cv_scores.std():.4f}")

    # 6. Save
    joblib.dump(model, "logistic_model.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    print("\n✅ Model and vectorizer saved.")

except Exception as e:
    print(f"❌ Error: {e}")