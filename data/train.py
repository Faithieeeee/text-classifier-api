import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_PATH = "data/bbc-text.csv"
MODEL_OUT_PATH = "models/classifier.joblib"
EMBEDDER_NAME = "all-MiniLM-L6-v2"

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df['category'].nunique()} classes")

    print("Loading embedding model:", EMBEDDER_NAME)
    embedder = SentenceTransformer(EMBEDDER_NAME)

    print("Encoding text (this can take a bit on CPU)...")
    X = embedder.encode(df["text"].tolist(), show_progress_bar=True)
    y = df["category"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training logistic regression classifier...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    print("\nEvaluation on held-out test set:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    print(f"Saving trained classifier to {MODEL_OUT_PATH}")
    joblib.dump(clf, MODEL_OUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
