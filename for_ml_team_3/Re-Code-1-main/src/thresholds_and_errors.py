import pandas as pd
import numpy as np
import joblib
import json
import re
import nltk
from sklearn.metrics import f1_score
from nltk.stem import WordNetLemmatizer

# Download WordNet (if first run)
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
LABELS = ["label_1", "label_2", "label_3", "label_4"]

# -----------------------------
# 1️⃣ Preprocessing
# -----------------------------
def lemmatize_text(text):
    """Lowercase, remove non-alpha, and lemmatize"""
    text = re.sub(r'[^\w\s]', ' ', str(text).lower())
    words = text.split()
    return ' '.join(lemmatizer.lemmatize(w) for w in words if w.isalpha())

def preprocess_texts(texts):
    """Apply lemmatization to a list/series of texts"""
    return [lemmatize_text(t) for t in texts if isinstance(t, str) and t.strip()]

# -----------------------------
# 2️⃣ Load data
# -----------------------------
def load_validation_data(path):
    df = pd.read_csv(path)
    X_texts = preprocess_texts(df["text"])
    y_true = df[LABELS].values
    return df, X_texts, y_true

# -----------------------------
# 3️⃣ Find optimal thresholds
# -----------------------------
def find_best_thresholds(model, X_transformed, y_true):
    """
    For each label, find the threshold that maximizes F1 score.
    Returns:
        thresholds (dict)
        probas (list of numpy arrays per label)
    """
    probas = model.predict_proba(X_transformed)  # list of arrays per label
    thresholds = {}

    for i, label in enumerate(LABELS):
        best_f1 = 0
        best_threshold = 0.5
        probs = probas[i][:, 1]  # probability of class 1

        for t in np.arange(0.1, 0.91, 0.05):
            preds = (probs >= t).astype(int)
            f1 = f1_score(y_true[:, i], preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        thresholds[label] = round(best_threshold, 2)

    return thresholds, probas

# -----------------------------
# 4️⃣ Collect mistakes
# -----------------------------
def collect_mistakes(raw_texts, y_true, probas, thresholds):
    """
    Returns a DataFrame of FP/FN for each label
    """
    rows = []
    for i, label in enumerate(LABELS):
        probs = probas[i][:, 1]
        preds = (probs >= thresholds[label]).astype(int)

        for idx, (true, pred, p) in enumerate(zip(y_true[:, i], preds, probs)):
            if true != pred:
                rows.append({
                    "text": raw_texts.iloc[idx],
                    "label": label,
                    "true": int(true),
                    "predicted": int(pred),
                    "probability": float(p),
                    "type": "FP" if pred == 1 else "FN"
                })

    return pd.DataFrame(rows)

# -----------------------------
# 5️⃣ Main execution
# -----------------------------
def main():
    # Paths - adjust if needed
    VAL_PATH = "data/dataset_C_val.csv"
    TRAIN_PATH = "data/dataset_C_train.csv"
    MODEL_PATH = "model.pkl"
    TFIDF_PATH = "tfidf.pkl"


    print("Loading validation data...")
    df_val, X_texts, y_true = load_validation_data(VAL_PATH)

    print("Loading model and TF-IDF...")
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)

    print("Transforming texts...")
    X_transformed = tfidf.transform(X_texts)

    print("Finding best thresholds per label...")
    thresholds, probas = find_best_thresholds(model, X_transformed, y_true)
    print("Thresholds:", thresholds)

    # Save thresholds
    with open("thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print("Saved thresholds.json")

    # Collect mistakes
    print("Collecting false positives / false negatives...")
    mistakes_df = collect_mistakes(df_val["text"], y_true, probas, thresholds)
    mistakes_df.to_csv("mistakes.csv", index=False)
    print("Saved mistakes.csv")

if __name__ == "__main__":
    main()
