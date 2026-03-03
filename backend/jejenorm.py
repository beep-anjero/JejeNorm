import re
import pickle
import os
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
#  LOAD SPACY MODEL
# ─────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# ─────────────────────────────────────────────
#  LEET-SPEAK MAP  (applied AFTER dictionary)
# ─────────────────────────────────────────────
LEET_MAP: Dict[str, str] = {
    '0': 'o',
    '1': 'i',
    '3': 'e',
    '4': 'a',
    '5': 's',
    '7': 't',
    '8': 'b',
    '9': 'g',
    '@': 'a',
}


def load_dataset() -> Dict[str, str]:
    """
    Load the Filipino slang / Jejemon normalization dictionary.
    Returns a dict mapping slang/abbreviated form → standard form.
    """
    print("Loading Filipino slang normalization dictionary...")
    return _build_rules()


def _build_rules() -> Dict[str, str]:
    return {

        # ── English internet abbreviations ────────────────────────────────
        'u':        'you',
        'ur':       'your',
        'uv':       'you have',
        'luv':      'love',
        'luve':     'love',
        'lv':       'love',
        'q':        'ko',

        # ── Numbers used as words ─────────────────────────────────────────
        '2':        'to',
        '4':        'for',
        '4ever':    'forever',
        '4eva':     'forever',
        '4evr':     'forever',
        '4evernmore': 'forever and ever',

        # ── Laughter & reactions ──────────────────────────────────────────
        'aw':       'aww',
        'yey':      'yay',
        'woe':      'wow',

        # ── Polite expressions ────────────────────────────────────────────
        'pls':      'please',
        'plz':      'please',
        'ty':       'thank you',
        'thnx':     'thanks',
        'thx':      'thanks',
        'sory':     'sorry',
        'sorc':     'sorry',
        'sori':     'sorry',

        # ── Agreement ────────────────────────────────────────────────────
        'yeah':     'yes',
        'yep':      'yes',
        'yup':      'yes',
        'nope':     'no',
        'nah':      'no',
        'kk':       'ok',
        'okie':     'ok',
        'okey':     'ok',
        'okay':     'ok',
        'okbye':    'ok bye',

        # ── Time / farewell abbreviations ────────────────────────────────
        'l8r':      'mamaya',
        'asap':     'bilisan',
        'gbye':     'bye',
        'g2g':      'kailangan umalis',
        'gtg':      'kailangan umalis',
        'brb':      'babalik kaagad',
        'bbl':      'babalik mamaya',
        'bbs':      'babalik agad',
        'tyl':      'hanggang mamaya',
        'ttyl':     'hanggang mamaya',
        'gmorning': 'magandang umaga',
        'gm':       'magandang umaga',
        'gnight':   'magandang gabi',
        'gn':       'magandang gabi',

        # ── Filipino particles ────────────────────────────────────────────
        'poe':      'po',
        'nalang':   'na lang',
        'nng':      'nang',

        # ── Jejemon spelling variants → standard Filipino ─────────────────
        'dIt':      'dito',
        'kta':      'kita',
        'tyo':      'tayo',
        'kht':      'kahit',
        'pRo':      'pero',
        'pro':      'pero',
        'mngyri':   'mangyari',
        'mngyare':  'mangyari',
        'mangyare': 'mangyari',
        'sayoh':    'sayo',
        'dhil':     'dahil',
        'dh2':      'dati',
        'lng':      'lang',
        'nmn':      'naman',
        'ulet':     'ulit',

        # ── Jejemon word variants ─────────────────────────────────────────
        'frnD':     'friend',
        'frnd':     'friend',
        'g0rl':     'girl',
        'gorl':     'girl',
        'gurl':     'girl',
        'hmw0rk':   'homework',
        'hmwork':   'homework',
        'sch00l':   'school',
        'rInGu':    'ring',
        'ringu':    'ring',

        # ── Communication ─────────────────────────────────────────────────
        'replyan':  'reply',
        'msg':      'message',
        'kwentong': 'kwento',
        'iikwen':   'kwento',

        # ── Abbreviations / contractions ──────────────────────────────────
        'dnt':      "don't",
        'dont':     "don't",
        'cant':     "can't",
        'wont':     "won't",
        'hav':      'have',
        'havnt':    "haven't",
        'didnt':    "didn't",
        'shouldve': 'should have',
        'couldve':  'could have',
        'wouldve':  'would have',
        'wanna':    'want to',
        'gonna':    'going to',
        'gotta':    'got to',
        'dunno':    "don't know",
        'duno':     "don't know",
        'idk':      "hindi ko alam",
        'kno':      'alam',
        'knw':      'alam',

        # ── Emotions / adjectives ─────────────────────────────────────────
        'gr8':      'great',
        'kl':       'cool',
        'nics':     'nice',
        'qute':     'cute',

        # ── Greetings ─────────────────────────────────────────────────────
        'hii':      'hi',
        'h3y':      'hey',
        'eow':      'hello',
        'ellow':    'hello',
        'kamuzta':  'kamusta',
        'kamustah': 'kamusta',

        # ── Filipino question words (abbreviated) ─────────────────────────
        'cnu':      'sino',
        'sinu':     'sino',
        'anong':    'ano',
        'anung':    'ano',

        # ── Filipino slang / colloquial ───────────────────────────────────
        'adyos':    'adios',
        'hangga':   'hanggang',
        'till':     'hanggang',
        'til':      'hanggang',
        'ulul':     'ulol',
        'sa yo':    'sayo',
    }


# ─────────────────────────────────────────────
#  VOCABULARY for fuzzy matching
# ─────────────────────────────────────────────
STANDARD_VOCABULARY = {
    'ako', 'ikaw', 'siya', 'kami', 'kayo', 'sila', 'tayo',
    'ko', 'mo', 'niya', 'namin', 'ninyo', 'nila',
    'ang', 'ng', 'sa', 'at', 'na', 'ay', 'pa', 'din', 'rin',
    'hindi', 'oo', 'opo', 'po', 'ba', 'nga', 'lang', 'naman',
    'sino', 'ano', 'saan', 'kailan', 'bakit', 'paano', 'kanino',
    'ito', 'iyan', 'iyon', 'dito', 'diyan', 'doon',
    'maganda', 'pangit', 'mabuti', 'masama', 'malaki', 'maliit',
    'mahal', 'mura', 'bago', 'luma', 'mabilis', 'mabagal',
    'kumain', 'matulog', 'maglaro', 'magsulat', 'magbasa',
    'pamilya', 'kaibigan', 'guro', 'estudyante', 'bahay', 'paaralan',
    'araw', 'gabi', 'umaga', 'tanghali', 'hapon',
    'salamat', 'sorry', 'kamusta', 'mabuhay',
    'sobra', 'talagang', 'talaga', 'kaya', 'kahit', 'dahil',
    'pero', 'kasi', 'para', 'kung', 'habang', 'pagkatapos',
    'sayo', 'kanila', 'namin', 'natin',
    'buhay', 'puso', 'isip', 'mata', 'ngiti', 'luha',
    'kwento', 'usap', 'tawag', 'sulat', 'sagot', 'tanong',
    'ulit', 'lagi', 'minsan', 'palagi', 'dati', 'ngayon', 'mamaya',
    'hanggang', 'mula', 'simula', 'wakas', 'huli',
    'love', 'hate', 'happy', 'sad', 'angry', 'tired',
    'friend', 'girl', 'boy', 'school', 'homework',
    'please', 'thanks', 'sorry', 'hello', 'bye',
    'ok', 'yes', 'no', 'maybe', 'same', 'cool', 'nice',
    'again', 'always', 'never', 'really', 'very', 'too',
}


def _fuzzy_correct_word(word: str, vocabulary: set = STANDARD_VOCABULARY, cutoff: float = 0.75) -> str:
    """Use edit-distance (difflib) to find the closest standard word."""
    if len(word) < 3:
        return word
    matches = get_close_matches(word, vocabulary, n=1, cutoff=cutoff)
    return matches[0] if matches else word


def _apply_leet(text: str) -> str:
    """Replace leet-speak characters with their letter equivalents."""
    for char, letter in LEET_MAP.items():
        text = text.replace(char, letter)
    return text


def _deduplicate_chars(text: str) -> str:
    """Collapse 3+ repeated characters to 2."""
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


# ─────────────────────────────────────────────
#  SPACY NLP PIPELINE
# ─────────────────────────────────────────────

def spacy_pipeline(text: str) -> str:
    """
    Apply SpaCy NLP pipeline to text:
    1. Tokenization
    2. Lemmatization
    3. Stop word removal
    4. Keep only NOUN and PROPN tokens (POS filtering)

    Returns a cleaned string of lemmatized content words.
    """
    doc = nlp(text)
    # Tokenize + lemmatize + remove stop words + filter by POS (NOUN, PROPN)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']
    ]
    return ' '.join(tokens)


def lower_replace(series: pd.Series) -> pd.Series:
    """
    Clean and normalize a Pandas Series of text:
    - Lowercase
    - Remove text inside brackets
    - Remove punctuation and special characters
    (Follows the text preprocessing lesson)
    """
    output = series.str.lower()
    output = output.str.replace(r'\[.*?\]', '', regex=True)
    output = output.str.replace(r'[^\w\s]', '', regex=True)
    return output


def token_lemma_nonstop(text: str) -> str:
    """
    Tokenize, lemmatize, and remove stop words from text using SpaCy.
    (Follows the SpaCy lesson)
    """
    doc = nlp(text)
    output = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(output)


def filter_pos(text: str, pos_list: list = ['NOUN', 'PROPN']) -> str:
    """
    Filter tokens by Part-of-Speech tag.
    Default: keep only nouns and proper nouns.
    (Follows the POS tagging lesson)
    """
    doc = nlp(text)
    output = [token.text for token in doc if token.pos_ in pos_list]
    return ' '.join(output)


def nlp_pipeline(series: pd.Series) -> pd.Series:
    """
    Full NLP preprocessing pipeline applied to a Pandas Series:
    1. Lowercase & clean text
    2. Tokenize + lemmatize + remove stop words
    3. Filter by POS
    (Follows the pipeline lesson)
    """
    output = lower_replace(series)
    output = output.apply(token_lemma_nonstop)
    output = output.apply(filter_pos)
    return output


# ─────────────────────────────────────────────
#  LABELED DATASET FOR ML TRAINING
# ─────────────────────────────────────────────

LABELED_DATA = [
    # Positive
    ("mahal kita sobra", "positive"),
    ("so happy today grabe ang saya", "positive"),
    ("love you forever friend", "positive"),
    ("ang ganda naman nito", "positive"),
    ("thank you so much salamat talaga", "positive"),
    ("best day ever so fun", "positive"),
    ("congrats grabe ang galing mo", "positive"),
    ("ang sweet naman niya", "positive"),
    ("happy birthday sana masaya ka", "positive"),
    ("love this so much ang cute", "positive"),
    ("amazing ang galing talaga", "positive"),
    ("so proud of you kaibigan", "positive"),
    ("beautiful place maganda talaga", "positive"),
    ("excited na sobra", "positive"),
    ("thank you always love you", "positive"),
    ("wonderful day with family", "positive"),
    ("great job lagi kang magaling", "positive"),
    ("smile always kasi maganda ka", "positive"),
    ("so blessed thankful talaga", "positive"),
    ("yay ganda ng balita", "positive"),
    ("I love you so much", "positive"),
    ("feeling happy today", "positive"),
    ("so excited for tomorrow", "positive"),
    ("you are the best friend", "positive"),
    ("life is beautiful and good", "positive"),
    ("i am so grateful today", "positive"),
    ("this is so wonderful", "positive"),
    ("you did amazing work today", "positive"),
    ("feeling blessed and happy", "positive"),
    ("great news today so happy", "positive"),

    # Negative
    ("galit na ako sa kanya", "negative"),
    ("ang pangit naman nito", "negative"),
    ("hate this so much", "negative"),
    ("sobrang sakit ng loob ko", "negative"),
    ("hindi ko na kaya", "negative"),
    ("disappointed sa nangyari", "negative"),
    ("ang sama naman niya", "negative"),
    ("toxic na tao yan", "negative"),
    ("ayoko na sobra na", "negative"),
    ("nakakainis talaga siya", "negative"),
    ("feel ko na bobo ako", "negative"),
    ("terrible day grabe ang pangit", "negative"),
    ("sad ako ngayon", "negative"),
    ("angry na talaga ako", "negative"),
    ("worst day ever", "negative"),
    ("so tired and exhausted na", "negative"),
    ("broken na ang puso ko", "negative"),
    ("stressed grabe na ang pagod", "negative"),
    ("depressed at sad ngayon", "negative"),
    ("fail na naman ako", "negative"),
    ("I hate this so much", "negative"),
    ("feeling so sad today", "negative"),
    ("this is terrible and bad", "negative"),
    ("so angry right now", "negative"),
    ("worst experience ever", "negative"),
    ("i am so tired and done", "negative"),
    ("this is so disappointing", "negative"),
    ("feeling broken and lost", "negative"),
    ("so stressed and exhausted", "negative"),
    ("bad day everything went wrong", "negative"),

    # Neutral
    ("kumain na ako kanina", "neutral"),
    ("pupunta ako bukas sa school", "neutral"),
    ("may pasok ba bukas", "neutral"),
    ("anong oras na", "neutral"),
    ("nasa bahay ako ngayon", "neutral"),
    ("mag-aaral muna ako", "neutral"),
    ("pababa na ako", "neutral"),
    ("saan ka pupunta bukas", "neutral"),
    ("ano ang schedule mo", "neutral"),
    ("natulog na ba siya", "neutral"),
    ("going to school now", "neutral"),
    ("just ate lunch today", "neutral"),
    ("what time is it na", "neutral"),
    ("i am at home now", "neutral"),
    ("going to sleep na", "neutral"),
    ("what is the schedule today", "neutral"),
    ("send me the file please", "neutral"),
    ("ok noted will do", "neutral"),
    ("see you later today", "neutral"),
    ("message me when you arrive", "neutral"),
    ("I am going to school", "neutral"),
    ("just finished eating dinner", "neutral"),
    ("what time does it start", "neutral"),
    ("i will be there later", "neutral"),
    ("ok i understand thank you", "neutral"),
    ("please send me the details", "neutral"),
    ("noted will reply soon", "neutral"),
    ("on my way now", "neutral"),
    ("will call you later today", "neutral"),
    ("just woke up now", "neutral"),
]


# ─────────────────────────────────────────────
#  ML SENTIMENT CLASSIFIER
# ─────────────────────────────────────────────

PICKLE_PATH = "sentiment_model.pkl"


def build_and_train_classifier():
    """
    Build and train Naive Bayes and Logistic Regression classifiers
    using TF-IDF vectorization on the labeled dataset.
    Saves the trained models as pickle files.
    (Follows the Naive Bayes + Logistic Regression lesson)
    """
    print("Building sentiment classifier...")

    # Create DataFrame from labeled data (follows lesson structure)
    df = pd.DataFrame(LABELED_DATA, columns=["text", "sentiment"])

    # Apply NLP pipeline: lowercase + clean
    df["text_clean"] = lower_replace(df["text"])
    df["text_clean"] = df["text_clean"].apply(token_lemma_nonstop)

    # TF-IDF Vectorization (follows lesson)
    tv = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = tv.fit_transform(df["text_clean"])
    y = df["sentiment"]

    # View features as DataFrame (follows lesson)
    X_df = pd.DataFrame(X.toarray(), columns=tv.get_feature_names_out())

    # Train/test split (follows lesson)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    # Naive Bayes model (follows lesson)
    model_nb = MultinomialNB()
    model_nb.fit(X_train, y_train)
    y_pred_nb = model_nb.predict(X_test)

    # Logistic Regression model (follows lesson)
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)

    # Print evaluation reports (follows lesson)
    print("\n=== Naive Bayes Results ===")
    print(classification_report(y_test, y_pred_nb))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")

    print("\n=== Logistic Regression Results ===")
    print(classification_report(y_test, y_pred_lr))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")

    # Save as pickle file (follows lesson)
    model_data = {
        "vectorizer": tv,
        "model_nb": model_nb,
        "model_lr": model_lr,
        "X_df_columns": list(tv.get_feature_names_out()),
    }
    pd.to_pickle(model_data, PICKLE_PATH)
    print(f"\nModels saved to {PICKLE_PATH}")

    return model_data


def load_or_train_classifier():
    """Load classifier from pickle if it exists, otherwise train and save it."""
    if os.path.exists(PICKLE_PATH):
        print("Loading saved sentiment model...")
        return pd.read_pickle(PICKLE_PATH)
    else:
        return build_and_train_classifier()


# Load classifier at module startup
_model_data = load_or_train_classifier()


def detect_sentiment_ml(text: str) -> Tuple[str, float]:
    """
    Detect sentiment using the trained Naive Bayes classifier with TF-IDF.
    Uses the full NLP pipeline: lowercase → clean → lemmatize → vectorize → predict.

    Returns:
        label (str)        – 'positive', 'negative', or 'neutral'
        confidence (float) – probability of predicted class (0.0–1.0)
    """
    tv = _model_data["vectorizer"]
    model_nb = _model_data["model_nb"]

    # Apply same preprocessing as training
    clean = text.lower()
    clean = re.sub(r'\[.*?\]', '', clean)
    clean = re.sub(r'[^\w\s]', '', clean)
    doc = nlp(clean)
    clean = ' '.join([token.lemma_ for token in doc if not token.is_stop])

    # Vectorize and predict
    X = tv.transform([clean])
    label = model_nb.predict(X)[0]
    proba = model_nb.predict_proba(X)[0]
    confidence = round(float(max(proba)), 2)

    return label, confidence


# ─────────────────────────────────────────────
#  JEJEMON NORMALIZATION PIPELINE
# ─────────────────────────────────────────────

def normalize_text(
    text: str,
    ngram_rules: Optional[Dict[str, str]] = None,
    use_fuzzy: bool = True,
) -> Tuple[str, List[dict]]:
    """
    Normalize Jejemon / Filipino internet slang text.

    Pipeline (in order):
      1. Lowercase
      2. Remove excessive punctuation clusters
      3. Deduplicate repeated characters
      4. Dictionary lookup
      5. Leet-speak conversion
      6. Fuzzy correction
      7. Clean up whitespace

    Returns:
      normalized (str)  – the cleaned output text
      diff (list[dict]) – per-word breakdown showing what changed and how
    """
    if ngram_rules is None:
        ngram_rules = load_dataset()

    normalized = text.lower()
    normalized = re.sub(r'([!?.,])\1+', r'\1', normalized)
    normalized = _deduplicate_chars(normalized)

    sorted_rules = sorted(ngram_rules.items(), key=lambda x: len(x[0]), reverse=True)
    for slang_word, standard_word in sorted_rules:
        pattern = r'(?<!\w)' + re.escape(slang_word.lower()) + r'(?!\w)'
        normalized = re.sub(pattern, standard_word, normalized, flags=re.IGNORECASE)

    normalized = _apply_leet(normalized)

    if use_fuzzy:
        tokens = normalized.split()
        corrected_tokens = []
        for token in tokens:
            stripped = token.strip(r"""!?.,;:'"()[]""")
            suffix = token[len(stripped):]
            prefix_len = len(token) - len(token.lstrip(r"""!?.,;:'"()[]"""))
            prefix = token[:prefix_len]
            stripped = token[prefix_len:len(token)-len(suffix)] if suffix else token[prefix_len:]

            if stripped and stripped not in STANDARD_VOCABULARY and stripped.isalpha():
                corrected = _fuzzy_correct_word(stripped)
                corrected_tokens.append(prefix + corrected + suffix)
            else:
                corrected_tokens.append(token)
        normalized = ' '.join(corrected_tokens)

    normalized = re.sub(r'\s+', ' ', normalized).strip()

    original_words = re.findall(r'\S+', text.lower())
    normalized_words = re.findall(r'\S+', normalized)
    diff = _build_diff(original_words, normalized_words)

    return normalized, diff


def _build_diff(original_words: List[str], normalized_words: List[str]) -> List[dict]:
    """Produce a simple word-level diff showing what changed."""
    diff = []
    max_len = max(len(original_words), len(normalized_words))
    for i in range(max_len):
        orig = original_words[i] if i < len(original_words) else ''
        norm = normalized_words[i] if i < len(normalized_words) else ''
        diff.append({
            'original': orig,
            'normalized': norm,
            'changed': orig != norm,
        })
    return diff


# ─────────────────────────────────────────────
#  RULE-BASED SENTIMENT (fallback)
# ─────────────────────────────────────────────

POSITIVE_WORDS = {
    'love', 'luv', 'mahal', 'amazing', 'awesome', 'great', 'wonderful',
    'fantastic', 'excellent', 'good', 'happy', 'joy', 'beautiful', 'maganda',
    'perfect', 'best', 'like', 'gusto', 'lol', 'haha', 'hihi', 'smile',
    'laugh', 'fun', 'cool', 'nice', 'brilliant', 'superb', 'adore',
    'gorgeous', 'lovely', 'delightful', 'terrific', 'stellar', 'saya',
    'masaya', 'mabuti', 'salamat', 'ganda', 'cute', 'sweet',
}

NEGATIVE_WORDS = {
    'hate', 'horrible', 'terrible', 'awful', 'bad', 'sad', 'angry',
    'upset', 'disappointed', 'disgusted', 'ugly', 'pangit', 'worst',
    'sucks', 'stupid', 'dumb', 'annoying', 'pathetic', 'miserable',
    'poor', 'fail', 'failed', 'sick', 'tired', 'exhausted', 'depressed',
    'broken', 'wrong', 'toxic', 'useless', 'worthless', 'disgusting',
    'galit', 'nakakainis', 'hayop', 'gago', 'bobo', 'tanga',
    'h8', 'sux', 'h4te',
}

NEGATION_WORDS = {'hindi', 'di', 'not', "don't", 'wala', 'ayaw', 'never'}


def detect_sentiment(text: str) -> Tuple[str, float]:
    """
    Detect sentiment using ML classifier (Naive Bayes + TF-IDF).
    Falls back to rule-based if ML fails.
    """
    try:
        return detect_sentiment_ml(text)
    except Exception:
        return _detect_sentiment_rulebased(text)


def _detect_sentiment_rulebased(text: str) -> Tuple[str, float]:
    """Rule-based sentiment fallback."""
    words = re.findall(r"\b\w[\w']*\b", text.lower())
    positive_score = 0
    negative_score = 0

    for i, word in enumerate(words):
        is_negated = i > 0 and words[i - 1] in NEGATION_WORDS
        if word in POSITIVE_WORDS:
            if is_negated:
                negative_score += 1
            else:
                positive_score += 1
        elif word in NEGATIVE_WORDS:
            if is_negated:
                positive_score += 1
            else:
                negative_score += 1

    total = positive_score + negative_score
    if total == 0:
        return 'neutral', 1.0

    confidence = round(max(positive_score, negative_score) / total, 2)
    if positive_score > negative_score:
        return 'positive', confidence
    elif negative_score > positive_score:
        return 'negative', confidence
    else:
        return 'neutral', 0.5


# ─────────────────────────────────────────────
#  EVALUATION UTILITIES
# ─────────────────────────────────────────────

def word_accuracy(predicted: str, reference: str) -> float:
    """Compute word-level accuracy between predicted and reference strings."""
    pred_words = predicted.lower().split()
    ref_words = reference.lower().split()
    if not ref_words:
        return 1.0
    length = max(len(pred_words), len(ref_words))
    matches = sum(
        1 for i in range(min(len(pred_words), len(ref_words)))
        if pred_words[i] == ref_words[i]
    )
    return round(matches / length, 4)


def normalization_rate(original: str, normalized: str) -> float:
    """Returns the proportion of words changed during normalization."""
    orig_words = original.lower().split()
    norm_words = normalized.lower().split()
    if not orig_words:
        return 0.0
    changed = sum(
        1 for i in range(min(len(orig_words), len(norm_words)))
        if orig_words[i] != norm_words[i]
    )
    return round(changed / len(orig_words), 4)