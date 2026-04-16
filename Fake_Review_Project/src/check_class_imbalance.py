"""
check_class_imbalance.py
Run this from your project root:
    python src/check_class_imbalance.py

What it does:
  1. Loads your training data
  2. Checks label distribution
  3. If imbalance worse than 60/40 → retrains with class_weight='balanced'
  4. Prints updated metrics
  5. Saves updated model as hybrid_model_balanced.pkl
"""

import joblib
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score,
                              confusion_matrix, classification_report)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re, os

analyzer = SentimentIntensityAnalyzer()

# ── Load data ─────────────────────────────────────────────────────────────────
# Adjust this path to wherever your processed training CSV lives
DATA_PATH = "data/processed/combined_dataset.csv"   # change if needed

print("=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    # Try common alternative paths
    for path in ["data/reviews_combined.csv", "reviews_combined.csv",
                 "data/processed/combined.csv", "data/combined.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            DATA_PATH = path
            break
    else:
        print("ERROR: Could not find training data CSV.")
        print("Please set DATA_PATH at the top of this script.")
        exit(1)

print(f"Loaded: {DATA_PATH}")
print(f"Shape:  {df.shape}")
print(f"Columns: {list(df.columns)}")

# Detect label column
label_col = None
for col in ["label", "Label", "fake", "deceptive", "target", "y"]:
    if col in df.columns:
        label_col = col
        break
if not label_col:
    label_col = df.columns[-1]
    print(f"No standard label column found — using last column: '{label_col}'")

# Detect text column
text_col = None
for col in ["review_text", "review", "text", "Text", "body", "comment"]:
    if col in df.columns:
        text_col = col
        break
if not text_col:
    text_col = df.select_dtypes(include="object").columns[0]
    print(f"No standard text column found — using: '{text_col}'")

print(f"\nText column:  '{text_col}'")
print(f"Label column: '{label_col}'")

# ── Step 2: Class distribution ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Class Distribution")
print("=" * 60)

counts   = df[label_col].value_counts()
total    = len(df)
minority = counts.min()
majority = counts.max()
ratio    = minority / majority

print(f"\nLabel counts:\n{counts.to_string()}")
print(f"\nMinority / Majority ratio: {ratio:.3f}  ({ratio*100:.1f}%)")

if ratio < 0.60:
    print(f"\n⚠  IMBALANCE DETECTED — ratio {ratio:.2f} is worse than 60/40")
    print("   → Will retrain with class_weight='balanced'")
    imbalanced = True
else:
    print(f"\n✓  Dataset is reasonably balanced ({ratio*100:.1f}%) — no fix required")
    print("   → Will still show comparison between default and balanced training")
    imbalanced = False

# ── Step 3: Feature helpers ───────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_stylometric(text):
    word_count = len(text.split())
    excl_count = text.count('!')
    caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
    sentences  = re.split(r'[.!?]', text)
    avg_sent   = word_count / max(1, len(sentences))
    fp_ratio   = sum(text.lower().count(p) for p in [' i ',' me ',' my ',' mine ']) / max(1, word_count)
    sent_int   = abs(analyzer.polarity_scores(text)['compound'])
    return [word_count, excl_count, caps_ratio, avg_sent, fp_ratio, sent_int]

# ── Step 4: Load existing model + vectorizer ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Loading existing model")
print("=" * 60)

tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("hybrid_model.pkl")
print("Loaded: hybrid_model.pkl + tfidf_vectorizer.pkl")

# ── Step 5: Prepare features ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Building feature matrix")
print("=" * 60)

texts  = df[text_col].fillna("").astype(str).tolist()
labels = df[label_col].tolist()

print("Cleaning text...")
cleaned = [clean_text(t) for t in texts]

print("Building TF-IDF vectors...")
X_tfidf = tfidf.transform(cleaned)

print("Extracting stylometric features...")
X_style = sp.csr_matrix(np.array([extract_stylometric(t) for t in texts]))

X = hstack([X_tfidf, X_style])
y = np.array(labels)
print(f"Feature matrix shape: {X.shape}")

# ── Step 6: Train/test split ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Step 7: Evaluate EXISTING model ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Existing model metrics (no class_weight)")
print("=" * 60)

y_pred_orig = model.predict(X_test)
y_prob_orig = model.predict_proba(X_test)[:, 1]

acc_orig  = accuracy_score(y_test, y_pred_orig)
auc_orig  = roc_auc_score(y_test, y_prob_orig)
f1_orig   = f1_score(y_test, y_pred_orig)
prec_orig = precision_score(y_test, y_pred_orig)
rec_orig  = recall_score(y_test, y_pred_orig)
cm_orig   = confusion_matrix(y_test, y_pred_orig)

print(f"Accuracy:  {acc_orig:.4f}")
print(f"ROC-AUC:   {auc_orig:.4f}")
print(f"F1-Score:  {f1_orig:.4f}")
print(f"Precision: {prec_orig:.4f}")
print(f"Recall:    {rec_orig:.4f}")
print(f"\nConfusion Matrix:\n{cm_orig}")
print(f"\nTN={cm_orig[0,0]}  FP={cm_orig[0,1]}")
print(f"FN={cm_orig[1,0]}  TP={cm_orig[1,1]}")

# ── Step 8: Retrain with class_weight='balanced' ──────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Retraining with class_weight='balanced'")
print("=" * 60)

model_balanced = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    C=1.0,
    solver='lbfgs',
    random_state=42
)
model_balanced.fit(X_train, y_train)
print("Training complete.")

y_pred_bal = model_balanced.predict(X_test)
y_prob_bal = model_balanced.predict_proba(X_test)[:, 1]

acc_bal  = accuracy_score(y_test, y_pred_bal)
auc_bal  = roc_auc_score(y_test, y_prob_bal)
f1_bal   = f1_score(y_test, y_pred_bal)
prec_bal = precision_score(y_test, y_pred_bal)
rec_bal  = recall_score(y_test, y_pred_bal)
cm_bal   = confusion_matrix(y_test, y_pred_bal)

print(f"Accuracy:  {acc_bal:.4f}")
print(f"ROC-AUC:   {auc_bal:.4f}")
print(f"F1-Score:  {f1_bal:.4f}")
print(f"Precision: {prec_bal:.4f}")
print(f"Recall:    {rec_bal:.4f}")
print(f"\nConfusion Matrix:\n{cm_bal}")
print(f"\nTN={cm_bal[0,0]}  FP={cm_bal[0,1]}")
print(f"FN={cm_bal[1,0]}  TP={cm_bal[1,1]}")

# ── Step 9: Cross-validation on balanced model ────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — 5-Fold Cross-Validation (balanced model)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model_balanced, X, y, cv=cv, scoring='accuracy')
print(f"Fold scores: {[round(s,4) for s in cv_scores]}")
print(f"CV Mean:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Step 10: Side-by-side comparison ─────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Metric':<15} {'Original':>12} {'Balanced':>12} {'Change':>10}")
print("-" * 52)
for metric, orig, bal in [
    ("Accuracy",  acc_orig,  acc_bal),
    ("ROC-AUC",   auc_orig,  auc_bal),
    ("F1-Score",  f1_orig,   f1_bal),
    ("Precision", prec_orig, prec_bal),
    ("Recall",    rec_orig,  rec_bal),
]:
    change = bal - orig
    arrow  = "▲" if change > 0 else ("▼" if change < 0 else "—")
    print(f"{metric:<15} {orig:>12.4f} {bal:>12.4f} {arrow}{abs(change):>8.4f}")

# ── Step 11: Decision and save ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — Decision")
print("=" * 60)

if imbalanced or f1_bal > f1_orig:
    joblib.dump(model_balanced, "hybrid_model_balanced.pkl")
    print("✓ Saved: hybrid_model_balanced.pkl")
    print("\nRECOMMENDATION: Use hybrid_model_balanced.pkl as your primary model.")
    print("Update dashboard.py line: joblib.load('hybrid_model_balanced.pkl')")
    if not imbalanced:
        print("NOTE: Dataset was balanced but class_weight='balanced' still improved F1.")
else:
    print("Original model performs equally or better — no change needed.")
    print("class_weight='balanced' is not required for this dataset.")

print("\n" + "=" * 60)
print("REPORT THIS IN YOUR CAPSTONE:")
print("=" * 60)
ratio_pct = f"{ratio*100:.1f}%"
print(f"""
Class Distribution Analysis:
  - Dataset size: {total:,} reviews
  - Label ratio (minority/majority): {ratio_pct}
  - Imbalance detected: {'Yes' if imbalanced else 'No'}
  - Handling: {'class_weight=balanced applied to LR' if imbalanced else 'No reweighting needed — dataset is sufficiently balanced'}
  - Impact on F1: {f1_orig:.4f} → {f1_bal:.4f} ({'improved' if f1_bal > f1_orig else 'no improvement'})

Confusion Matrix (balanced model, test set):
  TP={cm_bal[1,1]}  FP={cm_bal[0,1]}
  FN={cm_bal[1,0]}  TN={cm_bal[0,0]}
""")