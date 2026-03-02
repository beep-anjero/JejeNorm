import re
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple


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

    FIX: Removed duplicate keys, removed ambiguous single-letter entries
         (c, r, z, v, b, m, d, n) that caused false replacements in normal text,
         and removed raw number-only entries (0,1,3,4…) since those are now
         handled exclusively by the leet-speak conversion step AFTER dict lookup.
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
        'q':        'ko',        # Filipino slang: 'q' = 'ko'

        # ── Numbers used as words ─────────────────────────────────────────
        # (standalone, whole-word only)
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
        'nng':      'nang',         # lowercase of nNG

        # ── Jejemon spelling variants → standard Filipino ─────────────────
        'dIt':      'dito',         # case-folded below anyway
        'kta':      'kita',
        'tyo':      'tayo',
        'kht':      'kahit',
        'pRo':      'pero',         # case-folded; kept for clarity
        'pro':      'pero',
        'mngyri':   'mangyari',
        'mngyare':  'mangyari',
        'mangyare': 'mangyari',
        'sayoh':    'sayo',
        'dhil':     'dahil',
        'dh2':      'dati',
        'lng':      'lang',
        'nmn':      'naman',        # FIX: 'nmn' → 'naman' (not 'nalang')
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
        'eow':      'hello',        # classic Jejemon greeting
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
    # Common Filipino words
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
    # Common English words used in Filipino internet text
    'love', 'hate', 'happy', 'sad', 'angry', 'tired',
    'friend', 'girl', 'boy', 'school', 'homework',
    'please', 'thanks', 'sorry', 'hello', 'bye',
    'ok', 'yes', 'no', 'maybe', 'same', 'cool', 'nice',
    'again', 'always', 'never', 'really', 'very', 'too',
}


def _fuzzy_correct_word(word: str, vocabulary: set = STANDARD_VOCABULARY, cutoff: float = 0.75) -> str:
    """
    Use edit-distance (difflib) to find the closest standard word.
    Returns the original word if no close match is found.
    Only attempts correction on words of length >= 3 to avoid mangling particles.
    """
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
    """
    Collapse 3+ repeated characters to 2.
    e.g. 'sobrrrraaaa' → 'sobrra', 'hahahahaha' kept as-is (alternating)
    Specifically collapses runs: 'pleeeease' → 'pleease'
    """
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def normalize_text(
    text: str,
    ngram_rules: Optional[Dict[str, str]] = None,
    use_fuzzy: bool = True,
) -> Tuple[str, List[dict]]:
    """
    Normalize Jejemon / Filipino internet slang text.

    Pipeline (in order):
      1. Lowercase
      2. Remove excessive punctuation clusters (!!!!! → !)
      3. Deduplicate repeated characters (sobraaaa → sobraa)
      4. Dictionary lookup  ← slang forms with numbers/symbols looked up HERE
      5. Leet-speak conversion  ← numbers/symbols replaced AFTER dict lookup
      6. Fuzzy correction for unknown words (optional)
      7. Clean up whitespace

    Returns:
      normalized (str)  – the cleaned output text
      diff (list[dict]) – per-word breakdown showing what changed and how
    """
    if ngram_rules is None:
        ngram_rules = load_dataset()

    # ── Step 1: lowercase ────────────────────────────────────────────────
    normalized = text.lower()

    # ── Step 2: collapse punctuation runs ────────────────────────────────
    normalized = re.sub(r'([!?.,])\1+', r'\1', normalized)

    # ── Step 3: deduplicate characters ───────────────────────────────────
    normalized = _deduplicate_chars(normalized)

    # ── Step 4: dictionary lookup (BEFORE leet conversion) ───────────────
    # Sort by length descending so multi-word phrases match before single words
    sorted_rules = sorted(ngram_rules.items(), key=lambda x: len(x[0]), reverse=True)
    for slang_word, standard_word in sorted_rules:
        pattern = r'(?<!\w)' + re.escape(slang_word.lower()) + r'(?!\w)'
        normalized = re.sub(pattern, standard_word, normalized, flags=re.IGNORECASE)

  

FILIPINO_PHONETIC = [
    (r'\b2(\w+)', r'tu\1'),   # 2mama → tumama, 2naw → tunaw
    (r'\bc([aeiou])', r's\1'), # cya → sya, cnic → snic
    (r'\bng\b', 'nang'),       # standalone ng → nang
]

def apply_phonetic_rules(text: str) -> str:
    for pattern, replacement in FILIPINO_PHONETIC:
        text = re.sub(pattern, replacement, text)
    return text

    # ── Step 5: leet-speak conversion (AFTER dict lookup) ────────────────
    normalized = _apply_leet(normalized)

    # ── Step 6: fuzzy correction for remaining unknown-looking tokens ─────
    if use_fuzzy:
        tokens = normalized.split()
        corrected_tokens = []
        for token in tokens:
            # Strip punctuation for lookup, reattach afterward
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

    # ── Step 7: clean up whitespace ──────────────────────────────────────
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # ── Build word-level diff for the API response ────────────────────────
    original_words = re.findall(r'\S+', text.lower())
    normalized_words = re.findall(r'\S+', normalized)
    diff = _build_diff(original_words, normalized_words)

    return normalized, diff


def _build_diff(original_words: List[str], normalized_words: List[str]) -> List[dict]:
    """
    Produce a simple word-level diff showing what changed.
    Pairs words positionally; marks each as 'changed' or 'unchanged'.
    """
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
#  SENTIMENT DETECTION
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
    # Leet-speak negatives (checked before leet conversion on raw input)
    'h8', 'sux', 'h4te',
}

# Negation words that flip the next sentiment word
NEGATION_WORDS = {'hindi', 'di', 'not', "don't", 'wala', 'ayaw', 'never'}


def detect_sentiment(text: str) -> Tuple[str, float]:
    """
    Detect sentiment of the ORIGINAL (pre-normalization) text.

    FIX 1: Added neutral category when no sentiment words are found.
    FIX 2: Added basic negation handling (e.g. 'hindi masaya' → negative signal).
    FIX 3: Returns a confidence score (0.0–1.0) alongside the label.

    Returns:
        label (str)       – 'positive', 'negative', or 'neutral'
        confidence (float) – how confident the model is (0.0–1.0)
    """
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

    # ── FIX: Neutral when no sentiment signal found ───────────────────────
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
    """
    Compute word-level accuracy between predicted and reference strings.
    (% of word positions where prediction matches reference)
    """
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
    """
    Returns the proportion of words that were changed during normalization.
    Useful as a diagnostic metric.
    """
    orig_words = original.lower().split()
    norm_words = normalized.lower().split()
    if not orig_words:
        return 0.0
    changed = sum(
        1 for i in range(min(len(orig_words), len(norm_words)))
        if orig_words[i] != norm_words[i]
    )
    return round(changed / len(orig_words), 4)