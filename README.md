# JejeNorm
### An NLP-based Normalization System for Filipino Internet Slang and Jejemon Texts

---

## Overview

JejeNorm is a natural language processing (NLP) system that converts Filipino internet slang and Jejemon text into standard, readable form. It addresses the challenge of processing non-standard Filipino text found on social media platforms, which poses difficulties for downstream NLP tasks such as sentiment analysis, content moderation, and machine translation.

Jejemon is a writing style popularized in Filipino internet culture characterized by:
- Alternating letter cases (`hElLo pO`)
- Leet-speak substitutions (`h3y`, `s4yo`, `g0rl`)
- Phonetic abbreviations (`kht` = kahit, `dhil` = dahil)
- Repeated or extra characters (`sobrrraaa`, `plssss`)
- Mixed Filipino-English code-switching

---

## Features

- **Rule-based normalization** — regex patterns handle structural Jejemon features
- **Dictionary lookup** — 250+ curated Filipino slang and Jejemon word mappings
- **Leet-speak conversion** — character substitution decoding (`0→o`, `3→e`, `4→a`)
- **Fuzzy matching** — edit-distance correction for unknown words not in the dictionary
- **Sentiment detection** — classifies text as positive, negative, or neutral with confidence score and negation handling
- **Word-level diff** — highlights exactly which words were changed during normalization
- **REST API** — FastAPI backend with `/normalize` and `/evaluate` endpoints
- **Interactive frontend** — side-by-side web UI with sample Jejemon texts

---

## Project Structure

```
JejeNorm/
├── backend/
│   ├── jejenorm.py       # Core NLP normalization engine
│   ├── main.py           # FastAPI REST API
│   └── requirements.txt  # Python dependencies
├── frontend/
│   └── jejenorm.html     # Web interface
└── README.md
```

---

## Tech Stack

| Layer      | Technology                        |
|------------|-----------------------------------|
| Language   | Python 3.10+                      |
| Backend    | FastAPI, Uvicorn                  |
| NLP        | regex, difflib (edit distance)    |
| Frontend   | HTML, CSS, Vanilla JavaScript     |
| API Format | REST / JSON                       |

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/your-username/jejenorm.git
cd jejenorm
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install fastapi uvicorn
```

**4. Run the backend**
```bash
cd backend
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

**5. Open the frontend**

Open `frontend/jejenorm.html` in your browser. Make sure the backend is running first.

---

## API Reference

### `POST /normalize`

Normalizes a Jejemon or slang input text.

**Request body:**
```json
{
  "text": "H3y u!!! kamuzta nA? mIsS nA kTa sObRa!!!"
}
```

**Response:**
```json
{
  "original": "H3y u!!! kamuzta nA? mIsS nA kTa sObRa!!!",
  "normalized": "hey you! kamusta na? miss na kita sobra!",
  "sentiment": "positive",
  "sentiment_confidence": 0.75,
  "original_length": 7,
  "normalized_length": 8,
  "normalization_rate": 0.57,
  "words_changed": 4,
  "diff": [
    { "original": "h3y", "normalized": "hey", "changed": true },
    { "original": "u!!!", "normalized": "you!", "changed": true }
  ]
}
```

---

### `POST /evaluate`

Evaluates normalization accuracy against a gold-standard reference.

**Request body:**
```json
{
  "text": "H3y u!!! kamuzta nA?",
  "reference": "hey you! kamusta na?"
}
```

**Response:**
```json
{
  "original": "H3y u!!! kamuzta nA?",
  "normalized": "hey you! kamusta na?",
  "reference":  "hey you! kamusta na?",
  "word_accuracy": 1.0
}
```

---

### `GET /`
Health check — returns API status.

---

## Normalization Pipeline

Input text passes through the following stages in order:

```
1. Lowercase
        ↓
2. Collapse punctuation runs   (!!!!! → !)
        ↓
3. Deduplicate characters      (sobrrraaa → sobraa)
        ↓
4. Dictionary lookup           (h3y → hey, 4ever → forever, sch00l → school)
        ↓
5. Leet-speak conversion       (0→o, 1→i, 3→e, 4→a, 5→s, 7→t, @→a)
        ↓
6. Fuzzy matching              (edit distance for unknown words)
        ↓
7. Whitespace cleanup
```

> **Important:** Dictionary lookup runs *before* leet-speak conversion so that entries like `h3y` or `4ever` are matched as whole units before their characters are individually decoded.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Word Accuracy | % of output words that match the gold-standard reference |
| Normalization Rate | % of input words changed during normalization |
| Sentiment Confidence | Ratio of dominant sentiment words to total sentiment words detected |

---

## Sample Input / Output

| Input (Jejemon) | Output (Normalized) |
|---|---|
| `H3y u!!!` | `hey you!` |
| `kamuzta nA?` | `kamusta na?` |
| `lOvE u 4eVeR` | `love you forever` |
| `g4L1T 4K0 s4 iNyO` | `galit ako sa inyo` |
| `kHt aNoNg mNgYrI` | `kahit anong mangyari` |
| `S0rY i c@nt taLk` | `sorry i can't talk` |
| `pls rply nMn` | `please reply naman` |

---

## Limitations

- Dictionary coverage is limited to manually curated entries; novel Jejemon coinages may not be recognized
- Fuzzy matching may occasionally produce incorrect corrections for very short or ambiguous tokens
- Sentiment detection is lexicon-based and does not model context beyond immediate negation
- Code-switched sentences (Taglish) may produce mixed-language output

---

## Future Work

- Expand the dataset with crowd-sourced Jejemon–standard parallel pairs
- Fine-tune a sequence-to-sequence model (e.g., mBERT or RoBERTa-Tagalog) on the parallel corpus
- Implement BLEU score evaluation for more robust benchmarking
- Add part-of-speech tagging for better context-aware normalization

---

## Authors

- Jullian Anjelo Vidal
- Diether Manansala

---

## Acknowledgements

This project was developed as a course requirement for [Course Name] under [Professor's Name], [School Name].

---

## License

For academic use only.