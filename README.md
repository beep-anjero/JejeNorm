# JejeNorm

### An NLP-based Normalization System for Filipino Internet Slang and Jejemon Texts

JejeNorm converts Filipino internet slang and Jejemon-style text into a more standard, readable form. It combines a manually curated normalization lexicon, leet-speak decoding, fuzzy matching, and a simple sentiment classifier behind a FastAPI backend and a vanilla HTML/CSS/JavaScript frontend.

## Overview

Jejemon text commonly uses:

- Alternating letter cases, such as `hElLo pO`
- Leet-speak substitutions, such as `h3y`, `s4yo`, `g0rl`
- Phonetic abbreviations, such as `kht` for `kahit`
- Repeated or extra characters, such as `sobrrraaa`
- Filipino-English code-switching

This project does not depend on an external Jejemon database. Instead, it uses a manually curated rule-based lexicon inside the backend. That is acceptable for this project scope, but it also means new or unseen Jejemon variants may need to be added manually.

## Features

- Rule-based normalization for common Filipino internet slang and Jejemon forms
- Curated normalization lexicon with 120 current mappings
- Leet-speak conversion, such as `0 -> o`, `3 -> e`, `4 -> a`
- Fuzzy matching for unknown words using edit-distance matching
- Sentiment detection using TF-IDF and Naive Bayes
- Word-level diff for changed words
- FastAPI REST API with `/normalize` and `/evaluate` endpoints
- Browser-based frontend in plain HTML, CSS, and JavaScript

## Project Structure

```text
JejeNorm/
  backend/
    Dockerfile
    jejenorm.py
    main.py
    requirements.txt
    sentiment_model.pkl
  frontend/
    index.html
  deployment/
    huggingface/
      Dockerfile
      README.md
      jejenorm.py
      main.py
      requirements.txt
      sentiment_model.pkl
  tests/
    test_normalization.py
  README.md
```

`backend/` is the primary source of truth for development. `deployment/huggingface/` is a deployment copy for Hugging Face Spaces. When backend code changes, copy the updated backend files into `deployment/huggingface/` before redeploying.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn |
| NLP | regex, difflib, spaCy |
| ML | scikit-learn, pandas |
| Frontend | HTML, CSS, Vanilla JavaScript |
| API Format | REST / JSON |

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm
```

### Run the Backend

```bash
cd backend
uvicorn main:app --reload
```

The local API runs at:

```text
http://127.0.0.1:8000
```

### Open the Frontend

Open:

```text
frontend/index.html
```

The frontend currently uses the deployed API configured in `frontend/index.html`:

```js
const API = 'https://anjelow-jejenorm.hf.space';
```

For local testing, temporarily change it to:

```js
const API = 'http://127.0.0.1:8000';
```

## API Reference

### `POST /normalize`

Request:

```json
{
  "text": "g4L1T 4K0 s4 iNyO!!! pWeD3 xA kAyA hNdI kA pUmUNtA dIt??"
}
```

Response excerpt:

```json
{
  "normalized": "galit ako sa inyo! pwede siya kaya hindi ka pumunta dito?",
  "sentiment": "neutral",
  "sentiment_method": "Naive Bayes + TF-IDF"
}
```

### `POST /evaluate`

Request:

```json
{
  "text": "H3y u!!! kamuzta nA?",
  "reference": "hey you! kamusta na?"
}
```

Response excerpt:

```json
{
  "normalized": "hey you! kamusta na?",
  "word_accuracy": 1.0
}
```

### `GET /`

Health check endpoint.

## Normalization Pipeline

```text
1. Lowercase
2. Collapse repeated punctuation
3. Deduplicate repeated characters
4. Apply curated dictionary rules
5. Apply leet-speak conversion
6. Apply fuzzy correction
7. Clean whitespace
```

Dictionary lookup runs before leet-speak conversion so entries like `h3y` and `4ever` can be matched as whole words before their characters are decoded.

## Sample Input / Output

| Input | Output |
|---|---|
| `H3y u!!!` | `hey you!` |
| `kamuzta nA?` | `kamusta na?` |
| `lOvE u 4eVeR` | `love you forever` |
| `g4L1T 4K0 s4 iNyO` | `galit ako sa inyo` |
| `pWeD3 xA kAyA` | `pwede siya kaya` |
| `kHt aNoNg mNgYrI` | `kahit anong mangyari` |
| `S0rY i c@nt taLk` | `sorry i can't talk` |

## Testing

Run the normalization tests from the project root:

```bash
.venv\Scripts\python.exe -m unittest discover -s tests
```

## Reliability Notes

The normalization lexicon is manually curated, so it is reliable for covered forms but not complete for every possible Jejemon spelling. This is the correct tradeoff for a rule-based academic prototype because there is no single standard public Jejemon database that fully covers the project scope.

For presentation or defense, describe it this way:

```text
We did not use an existing Jejemon database because there is no reliable standard dataset for our exact scope. Instead, we created a manually curated normalization lexicon and combined it with leet-speak decoding and fuzzy matching.
```

## Limitations

- New Jejemon variants may need manual rule additions
- Fuzzy matching can miscorrect short or ambiguous tokens
- Sentiment classification is trained on a small local dataset
- Code-switched Taglish can produce mixed-language normalized output

## Future Work

- Move normalization rules from Python into `backend/data/normalization_rules.json` or `.csv`
- Expand the lexicon using crowd-sourced Jejemon-standard pairs
- Add a larger gold-standard evaluation set
- Add BLEU or character-level edit distance metrics
- Improve sentiment modeling with a larger Filipino/Taglish dataset

## Authors

- Jullian Anjelo Vidal
- Diether Manansala

## License

For academic use only.
