from fastapi import FastAPI
from pydantic import BaseModel
from jejenorm import normalize_text, detect_sentiment, load_dataset, word_accuracy, normalization_rate
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="JejeNorm API", version="2.0")

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dictionary once at startup
ngram_rules = load_dataset()


class TextInput(BaseModel):
    text: str


class EvalInput(BaseModel):
    """For evaluation endpoint: provide input text and a gold-standard reference."""
    text: str
    reference: str


@app.post("/normalize")
def normalize(input: TextInput):
    """
    Normalize Jejemon/slang text.
    Returns the normalized text, sentiment, confidence, word diff, and stats.
    """
    normalized, diff = normalize_text(input.text, ngram_rules)
    sentiment, confidence = detect_sentiment(input.text)

    changed_words = [d for d in diff if d['changed']]

    return {
        "original":          input.text,
        "normalized":        normalized,
        "sentiment":         sentiment,
        "sentiment_confidence": confidence,
        "original_length":   len(input.text.split()),
        "normalized_length": len(normalized.split()),
        "normalization_rate": normalization_rate(input.text, normalized),
        "words_changed":     len(changed_words),
        "diff":              diff,
    }


@app.post("/evaluate")
def evaluate(input: EvalInput):
    """
    Evaluate normalization quality against a gold-standard reference.
    Useful for testing and academic reporting.
    """
    normalized, diff = normalize_text(input.text, ngram_rules)
    accuracy = word_accuracy(normalized, input.reference)

    return {
        "original":   input.text,
        "normalized": normalized,
        "reference":  input.reference,
        "word_accuracy": accuracy,
    }


@app.get("/")
def root():
    return {"message": "JejeNorm API v2.0 is running!"}