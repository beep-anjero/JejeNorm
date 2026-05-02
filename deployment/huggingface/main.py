from fastapi import FastAPI
from pydantic import BaseModel
from jejenorm import (
    normalize_text, detect_sentiment, load_dataset,
    word_accuracy, normalization_rate,
    spacy_pipeline, nlp_pipeline, _model_data
)
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI(title="JejeNorm API", version="3.0")

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
    text: str
    reference: str


@app.post("/normalize")
def normalize(input: TextInput):
    """
    Normalize Jejemon/slang text and analyze sentiment.

    Pipeline:
    1. JejeNorm normalization (dictionary + leet + fuzzy)
    2. SpaCy NLP pipeline (tokenize → lemmatize → stop word removal → POS filter)
    3. TF-IDF vectorization
    4. Naive Bayes sentiment classification
    """
    # Step 1: Normalize Jejemon text
    normalized, diff = normalize_text(input.text, ngram_rules)

    # Step 2: Apply SpaCy pipeline to normalized text
    spacy_processed = spacy_pipeline(normalized)

    # Step 3 & 4: ML sentiment classification (TF-IDF + Naive Bayes)
    sentiment, confidence = detect_sentiment(input.text)

    changed_words = [d for d in diff if d['changed']]

    return {
        "original":               input.text,
        "normalized":             normalized,
        "spacy_processed":        spacy_processed,
        "sentiment":              sentiment,
        "sentiment_confidence":   confidence,
        "sentiment_method":       "Naive Bayes + TF-IDF",
        "original_length":        len(input.text.split()),
        "normalized_length":      len(normalized.split()),
        "normalization_rate":     normalization_rate(input.text, normalized),
        "words_changed":          len(changed_words),
        "diff":                   diff,
    }


@app.post("/evaluate")
def evaluate(input: EvalInput):
    """
    Evaluate normalization quality against a gold-standard reference.
    """
    normalized, diff = normalize_text(input.text, ngram_rules)
    accuracy = word_accuracy(normalized, input.reference)

    return {
        "original":      input.text,
        "normalized":    normalized,
        "reference":     input.reference,
        "word_accuracy": accuracy,
    }


@app.get("/model-info")
def model_info():
    """
    Returns information about the trained ML models.
    Shows the NLP pipeline used: TF-IDF + Naive Bayes + Logistic Regression.
    """
    tv = _model_data["vectorizer"]
    return {
        "pipeline": [
            "1. Lowercase & clean text (Pandas)",
            "2. Tokenization (SpaCy)",
            "3. Lemmatization (SpaCy)",
            "4. Stop word removal (SpaCy)",
            "5. POS filtering (SpaCy - NOUN, PROPN, VERB, ADJ)",
            "6. TF-IDF Vectorization (Scikit-learn)",
            "7. Naive Bayes Classification (Scikit-learn)",
            "8. Logistic Regression Classification (Scikit-learn)",
        ],
        "vectorizer":        "TfidfVectorizer",
        "ngram_range":       "(1, 2)",
        "vocabulary_size":   len(tv.get_feature_names_out()),
        "models_trained":    ["MultinomialNB", "LogisticRegression"],
        "sentiment_classes": ["positive", "negative", "neutral"],
        "pickle_file":       "sentiment_model.pkl",
    }


@app.get("/")
def root():
    return {"message": "JejeNorm API v3.0 is running!"}