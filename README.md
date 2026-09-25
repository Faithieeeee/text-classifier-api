 ## Text Classifier API
A backend API that classifies news text into one of five categories: business, entertainment, politics, sport, or tech.
 How it works
- Text is converted into numerical embeddings using a pretrained sentence-transformer model (`all-MiniLM-L6-v2`)
- A logistic regression classifier predicts the category from those embeddings
- Achieves 98% accuracy on held-out test data
## Tech stack
- **FastAPI** — backend API framework
- **sentence-transformers** — text embedding
- **scikit-learn** — classifier
- **Python**
## Endpoints
- `GET /health` — check the service is running
- `POST /predict` — classify a single piece of text
- `POST /batch-predict` — classify multiple texts at once

Both prediction endpoints require an `x-api-key` header.
## Running locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then visit `http://localhost:8000/docs` for interactive API docs.
