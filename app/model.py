import joblib
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "classifier.joblib"
EMBEDDER_NAME = "all-MiniLM-L6-v2"

class ClassifierService:
    """
      Wraps the embedder + classifier so they're loaded once when the API
    starts
    """

    def __init__(self):
        self.embedder = None
        self.clf = None

    def load(self):
        print("Loading embedder...")
        self.embedder = SentenceTransformer(EMBEDDER_NAME)
        print("Loading classifier from", MODEL_PATH)
        self.clf = joblib.load(MODEL_PATH)
        print("Model service ready.")

    @property
    def is_loaded(self) -> bool:
        return self.clf is not None and self.embedder is not None

    def predict(self, text: str):
        embedding = self.embedder.encode([text])
        label = self.clf.predict(embedding)[0]
        proba = self.clf.predict_proba(embedding)[0]
        confidence = float(max(proba))
        return label, confidence

    def predict_batch(self, texts: list[str]):
        embeddings = self.embedder.encode(texts)
        labels = self.clf.predict(embeddings)
        probas = self.clf.predict_proba(embeddings)
        return [
            (label, float(max(proba)))
            for label, proba in zip(labels, probas)
        ]


# one shared instance the rest of the app will import and use
classifier_service = ClassifierService()