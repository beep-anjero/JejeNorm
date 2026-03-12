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
    # ── POSITIVE (150) ───────────────────────────────────────────────────────
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
    ("sobrang saya ko ngayon talaga", "positive"),
    ("ang bait niya grabe", "positive"),
    ("napaka galing mo talaga", "positive"),
    ("masaya ako sa nangyari", "positive"),
    ("ang sarap ng pagkain namin", "positive"),
    ("grabe ang ganda ng lugar na ito", "positive"),
    ("naging maayos ang lahat salamat", "positive"),
    ("lagi kang nasa puso ko", "positive"),
    ("ang daming magagandang bagay ngayon", "positive"),
    ("so grateful to have you in my life", "positive"),
    ("congrats sa iyong tagumpay", "positive"),
    ("ang saya saya namin kanina", "positive"),
    ("masayang masaya ang aking pamilya", "positive"),
    ("ang galing ng performance mo", "positive"),
    ("nag enjoy kami sobra kanina", "positive"),
    ("fulfilled ako sa aking ginawa", "positive"),
    ("ang ganda ng buhay pag may kasama", "positive"),
    ("so in love with everything right now", "positive"),
    ("feel ko ang pagmamahal ng lahat", "positive"),
    ("ang swerte ko sa mga taong nasa buhay ko", "positive"),
    ("you make me so happy every day", "positive"),
    ("this place is so amazing and peaceful", "positive"),
    ("so thankful for this opportunity", "positive"),
    ("best decision i ever made", "positive"),
    ("i feel so loved and appreciated", "positive"),
    ("today was absolutely perfect", "positive"),
    ("so happy to see you again", "positive"),
    ("you always make me smile", "positive"),
    ("feeling so motivated and inspired", "positive"),
    ("ang sarap mabuhay pag masaya ka", "positive"),
    ("so glad everything worked out well", "positive"),
    ("ang ganda ng simula ng araw ko", "positive"),
    ("napaka swerte ko talaga ngayon", "positive"),
    ("ang init ng pagtanggap sa akin", "positive"),
    ("sobrang thankful sa lahat ng pagpapala", "positive"),
    ("ang galing galing talaga niya", "positive"),
    ("naging matagumpay ang aming proyekto", "positive"),
    ("so proud ng aming grupo ngayon", "positive"),
    ("ang saya ng bawat sandali kasama kayo", "positive"),
    ("grabe ang sarap ng feeling na ito", "positive"),
    ("you are truly an inspiration to me", "positive"),
    ("this song makes me so happy", "positive"),
    ("i am so lucky to have this life", "positive"),
    ("ang ganda ng weather today perfect", "positive"),
    ("everything is falling into place nicely", "positive"),
    ("so excited about the future ahead", "positive"),
    ("you did a fantastic job today", "positive"),
    ("napaka ganda ng aming samahan", "positive"),
    ("ang bait ng lahat ng tao dito", "positive"),
    ("sobrang saya ng birthday celebration namin", "positive"),
    ("ang daming nakuha naming blessings", "positive"),
    ("feel so alive and full of energy", "positive"),
    ("this is the best thing ever happened", "positive"),
    ("so happy with how things turned out", "positive"),
    ("ang saya pag kasama ang pamilya", "positive"),
    ("i love spending time with you", "positive"),
    ("you bring so much joy into my life", "positive"),
    ("ang galing mo talagang kumanta", "positive"),
    ("so fulfilled after finishing this project", "positive"),
    ("ang sarap ng tagumpay na pinaghirapan", "positive"),
    ("best feeling ever to achieve your goal", "positive"),
    ("ang saya ng makapag rest pagkatapos ng trabaho", "positive"),
    ("so inspired by your story kaibigan", "positive"),
    ("napaka supportive ng aking mga kaibigan", "positive"),
    ("ang ganda ng aming friendship talaga", "positive"),
    ("so happy na natulungan kita ngayon", "positive"),
    ("ang warm ng pakiramdam ko kasama kayo", "positive"),
    ("feeling great and unstoppable today", "positive"),
    ("so thankful for every little blessing", "positive"),
    ("you are the reason i smile every day", "positive"),
    ("ang sarap ng matulog nang maayos", "positive"),
    ("sobrang saya ko sa aking bagong trabaho", "positive"),
    ("ang ganda ng surprise na ginawa nila", "positive"),
    ("so happy na naabot ko na ang pangarap ko", "positive"),
    ("ang bait ng aking bagong kaklase", "positive"),
    ("everything feels right today", "positive"),
    ("so grateful for this beautiful life", "positive"),
    ("ang daming dahilan para magpasalamat", "positive"),
    ("you are such a wonderful person", "positive"),
    ("feeling so content and at peace", "positive"),
    ("ang sarap ng mahal at mahalin", "positive"),
    ("so happy seeing my family happy", "positive"),
    ("this experience is truly unforgettable", "positive"),
    ("ang ganda ng moments na ito", "positive"),
    ("so proud of how far i have come", "positive"),
    ("napaka ganda ng buhay pag positibo ka", "positive"),
    ("ang saya ng makatanggap ng magandang balita", "positive"),
    ("you always know how to cheer me up", "positive"),
    ("so blessed to wake up every morning", "positive"),
    ("ang init ng puso ng aking pamilya", "positive"),
    ("feeling so happy and complete today", "positive"),
    ("so excited to start this new chapter", "positive"),
    ("ang galing ng team namin sobra", "positive"),
    ("napaka masaya ng aming reunion kanina", "positive"),
    ("so glad i chose to be positive today", "positive"),
    ("ang sarap ng feeling ng accomplishment", "positive"),
    ("you inspire me to be a better person", "positive"),
    ("so happy for your success kaibigan", "positive"),
    ("ang magandang bagay ay darating din", "positive"),
    ("feeling so joyful and grateful right now", "positive"),
    ("ang saya ng matupad ang iyong pangarap", "positive"),
    ("so thankful for your kind words today", "positive"),
    ("ang ganda ng aming bonding kasama pamilya", "positive"),
    ("you mean the world to me talaga", "positive"),
    ("so happy na kumpleto na ang aming grupo", "positive"),
    ("ang sarap ng maging masaya kahit maliliit na bagay", "positive"),
    ("feeling loved and surrounded by good people", "positive"),
    ("so excited for what is coming next", "positive"),
    ("ang galing mo lagi at hanga ako sayo", "positive"),
    ("napaka saya ng araw na ito", "positive"),
    ("so happy and thankful for everything", "positive"),
    ("ang daming pagmamahal na natatanggap ko", "positive"),
    ("you are such a blessing in my life", "positive"),

    # ── NEGATIVE (150) ───────────────────────────────────────────────────────
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
    ("ayaw ko na dito grabe", "negative"),
    ("hindi na ako makapagtiis", "negative"),
    ("sobrang hirap ng pinagdadaan ko", "negative"),
    ("walang nagmamalasakit sa akin", "negative"),
    ("nakakaiyak ang lahat ng nangyayari", "negative"),
    ("ang daming problema hindi ko na kaya", "negative"),
    ("grabe ang sakit ng tinaggap ko", "negative"),
    ("pakiramdam ko wala akong halaga", "negative"),
    ("ang hirap mabuhay na ganito", "negative"),
    ("sobrang lungkot ko talaga ngayon", "negative"),
    ("napaka unfair ng buhay", "negative"),
    ("parang lahat ay laban sa akin", "negative"),
    ("wala na akong lakas para lumaban", "negative"),
    ("ang sakit ng pagtataksil na ito", "negative"),
    ("hindi ko mapatawad ang nangyari", "negative"),
    ("so done with everything right now", "negative"),
    ("this situation is making me so anxious", "negative"),
    ("i feel so alone and misunderstood", "negative"),
    ("nothing ever goes right for me", "negative"),
    ("so frustrated with everything today", "negative"),
    ("i cannot take this anymore it hurts", "negative"),
    ("feeling so hopeless and empty inside", "negative"),
    ("ang hirap ng maging malayo sa mahal mo sa buhay", "negative"),
    ("sobrang lungkot ng pakiramdam ko ngayon", "negative"),
    ("grabe ang bigat ng nararamdaman ko", "negative"),
    ("ang sakit ng mawalan ng mahal mo sa buhay", "negative"),
    ("hindi ko alam kung anong gagawin ko", "negative"),
    ("pakiramdam ko nag iisa lang ako sa mundo", "negative"),
    ("sobrang pagod na ako sa lahat", "negative"),
    ("ang hirap ng hindi maintindihan ng mga tao", "negative"),
    ("nakakainis na ang lahat ng nangyayari", "negative"),
    ("this is the worst thing that happened to me", "negative"),
    ("i feel so betrayed and hurt right now", "negative"),
    ("so disappointed in the people around me", "negative"),
    ("everything feels so heavy and overwhelming", "negative"),
    ("i hate how things turned out today", "negative"),
    ("so upset and cannot calm down", "negative"),
    ("ang sama ng loob ko sa kanya", "negative"),
    ("hindi ko deserve ang ganitong trato", "negative"),
    ("sobrang sakit na hindi ko na kayang tiisin", "negative"),
    ("ang daming pinagdaanan ko at wala pang nagbabago", "negative"),
    ("feeling so broken after what happened", "negative"),
    ("so angry and cannot stop crying", "negative"),
    ("this pain is unbearable right now", "negative"),
    ("i feel like giving up on everything", "negative"),
    ("so exhausted from trying so hard", "negative"),
    ("ang hirap ng maging malakas para sa lahat", "negative"),
    ("sobrang stress ko at hindi ko na kaya", "negative"),
    ("pakiramdam ko palagi na lang akong nagkakamali", "negative"),
    ("ang sakit ng maging hindi sapat", "negative"),
    ("hindi ko makuha ang gusto ko kahit gaano ko subukan", "negative"),
    ("so tired of being strong all the time", "negative"),
    ("i feel so worthless and unimportant", "negative"),
    ("ang hirap ng pagtanggap ng katotohanan", "negative"),
    ("sobrang galit ko sa sarili ko", "negative"),
    ("grabe ang lungkot na nararamdaman ko", "negative"),
    ("ang sakit ng mawala ang tiwala mo sa isang tao", "negative"),
    ("feeling so drained and unmotivated today", "negative"),
    ("so sad that things did not work out", "negative"),
    ("i hate feeling this way all the time", "negative"),
    ("ang hirap ng maging masaya pag malungkot ka", "negative"),
    ("sobrang pagod na ako sa pag aayos ng lahat", "negative"),
    ("everything around me feels so negative", "negative"),
    ("i feel like nobody understands me", "negative"),
    ("so hurt by the words they said to me", "negative"),
    ("ang sakit ng pakiramdam na iniwan ka", "negative"),
    ("hindi ko matanggap ang nangyari sa amin", "negative"),
    ("so overwhelmed and cannot focus at all", "negative"),
    ("feeling so anxious and scared right now", "negative"),
    ("ang hirap ng mag move on sa masasakit na karanasan", "negative"),
    ("sobrang lungkot ko pag naiisip ko ang nakaraan", "negative"),
    ("i am so disappointed in myself today", "negative"),
    ("ang sama ng pakiramdam ko pagkagising ko", "negative"),
    ("so defeated and nothing seems to work", "negative"),
    ("i hate that i keep failing over and over", "negative"),
    ("ang sakit ng hindi mapansin ng taong mahal mo", "negative"),
    ("sobrang hirap ng tanggapin na tapos na ang lahat", "negative"),
    ("feeling so low and cannot get back up", "negative"),
    ("so angry at the situation i am in", "negative"),
    ("i feel like i am not good enough", "negative"),
    ("ang hirap ng maging sapat sa mata ng iba", "negative"),
    ("sobrang pagod na ako sa lahat ng drama", "negative"),
    ("this is all so unfair and painful", "negative"),
    ("so sad and do not know how to feel better", "negative"),
    ("ang sakit ng mapagsamantalahan ng taong tiwala mo", "negative"),
    ("feeling so empty and numb inside", "negative"),
    ("so frustrated nothing ever changes", "negative"),
    ("i feel so lost and confused right now", "negative"),
    ("ang hirap ng pakiramdam na hindi ka mahal", "negative"),
    ("sobrang sakit ng karanasang ito", "negative"),
    ("so heartbroken and do not know what to do", "negative"),
    ("i hate that i care so much and get nothing back", "negative"),
    ("ang lungkot ng maging malayo sa mga mahal mo sa buhay", "negative"),
    ("sobrang galit ko sa mga nangyari ngayon", "negative"),
    ("feeling so miserable and hopeless today", "negative"),
    ("so tired of being hurt over and over again", "negative"),
    ("i feel so neglected and taken for granted", "negative"),
    ("ang sakit ng mapabayaan ng taong pinagkakatiwalaan mo", "negative"),
    ("sobrang hirap ng maging malakas kahit nasasaktan ka", "negative"),
    ("this day was absolutely terrible and draining", "negative"),
    ("so angry and hurt and do not want to talk", "negative"),
    ("i feel like the world is against me", "negative"),
    ("ang daming negatibong nangyayari sa buhay ko", "negative"),
    ("sobrang lungkot at pagod na ako sa lahat", "negative"),
    ("feeling completely broken and defeated today", "negative"),
    ("so sad that everything fell apart again", "negative"),
    ("i hate this feeling so much right now", "negative"),

    # ── NEUTRAL (150) ────────────────────────────────────────────────────────
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
    ("nandito na ako sa opisina", "neutral"),
    ("ilang oras pa bago matapos", "neutral"),
    ("nagpadala na ako ng file", "neutral"),
    ("pwede ba nating i reschedule ang meeting", "neutral"),
    ("nagluto na ako kanina para sa hapunan", "neutral"),
    ("nakita ko siya kanina sa mall", "neutral"),
    ("kahapon lang namin napag usapan iyon", "neutral"),
    ("may klase kami bukas ng hapon", "neutral"),
    ("naka lock ba ang pintuan", "neutral"),
    ("ililipat natin ang meeting sa Martes", "neutral"),
    ("sinend ko na ang email kanina", "neutral"),
    ("anong oras ang iyong klase bukas", "neutral"),
    ("bibili muna ako ng pagkain bago umuwi", "neutral"),
    ("naka charge na ba ang laptop ko", "neutral"),
    ("mag papasa kami ng requirements bukas", "neutral"),
    ("i already submitted the assignment online", "neutral"),
    ("the meeting is scheduled for three pm", "neutral"),
    ("please check your email for the details", "neutral"),
    ("i will finish this later tonight", "neutral"),
    ("just checking in to see if you are ok", "neutral"),
    ("can you send me the link please", "neutral"),
    ("the class starts at eight in the morning", "neutral"),
    ("i need to buy groceries after school", "neutral"),
    ("please remind me about the deadline tomorrow", "neutral"),
    ("i already read the instructions carefully", "neutral"),
    ("the report is due on friday", "neutral"),
    ("i will be online in a few minutes", "neutral"),
    ("let me know when you are ready", "neutral"),
    ("the store closes at nine in the evening", "neutral"),
    ("i have a meeting at two this afternoon", "neutral"),
    ("just downloaded the app on my phone", "neutral"),
    ("can we meet at the library tomorrow", "neutral"),
    ("i will bring the materials to class", "neutral"),
    ("the event starts at six in the evening", "neutral"),
    ("please read chapter three for next session", "neutral"),
    ("i already saved the document on my drive", "neutral"),
    ("the bus arrives every thirty minutes", "neutral"),
    ("i need to print my assignment tonight", "neutral"),
    ("naligo na ako at handa na para sa klase", "neutral"),
    ("nag text na si teacher tungkol sa assignment", "neutral"),
    ("bukas ang submission ng requirements", "neutral"),
    ("ililipat ko ang gamit ko sa ibang kwarto", "neutral"),
    ("kakain muna kami bago pumunta", "neutral"),
    ("naka park na ako sa harap ng building", "neutral"),
    ("ilang minuto na lang at darating na sila", "neutral"),
    ("puwede mo ba akong i pick up mamaya", "neutral"),
    ("nag update na ang app sa aking telepono", "neutral"),
    ("nasa daan pa lang kami papunta doon", "neutral"),
    ("the assignment needs to be submitted by midnight", "neutral"),
    ("i am currently working on the project", "neutral"),
    ("please wait for further instructions", "neutral"),
    ("the document has been uploaded already", "neutral"),
    ("i will check the schedule and get back to you", "neutral"),
    ("the presentation is set for next week", "neutral"),
    ("i have three classes today back to back", "neutral"),
    ("just received the confirmation email", "neutral"),
    ("please fill out the form before Friday", "neutral"),
    ("i need to review my notes before the exam", "neutral"),
    ("the library is open until ten tonight", "neutral"),
    ("i will forward the message to the group", "neutral"),
    ("just logged in to the online portal", "neutral"),
    ("the requirements are listed on the syllabus", "neutral"),
    ("i need to ask the teacher about the topic", "neutral"),
    ("we have a group study session tomorrow", "neutral"),
    ("naka print na ba ang aming handouts", "neutral"),
    ("nasa canteen kami ngayon kumakain", "neutral"),
    ("pumunta na kami sa classroom namin", "neutral"),
    ("hinihintay namin ang guro namin ngayon", "neutral"),
    ("nag search na ako ng sagot sa tanong", "neutral"),
    ("kasama ko ang grupo ko sa library", "neutral"),
    ("mag submit na kami ng project bukas", "neutral"),
    ("nagpadala na ako ng message sa teacher", "neutral"),
    ("naka assign ako sa part na ito ng project", "neutral"),
    ("kinopya ko na ang notes mula sa board", "neutral"),
    ("the teacher explained the lesson clearly today", "neutral"),
    ("i took note of everything discussed in class", "neutral"),
    ("the exam covers chapters one to five", "neutral"),
    ("i need to buy a new notebook for class", "neutral"),
    ("the group decided to meet on Saturday", "neutral"),
    ("please share the presentation file with me", "neutral"),
    ("i will attend the seminar tomorrow morning", "neutral"),
    ("the deadline was moved to next week", "neutral"),
    ("i already paid the school fees online", "neutral"),
    ("please confirm your attendance by Thursday", "neutral"),
    ("nag aabang na kami ng jeep papunta sa school", "neutral"),
    ("kailangan kong pumunta sa admin office bukas", "neutral"),
    ("naka schedule na ang aming defense next month", "neutral"),
    ("ipinasa ko na ang aking enrollment form", "neutral"),
    ("nagpunta kami sa computer lab para sa project", "neutral"),
    ("nag download na ako ng reference materials", "neutral"),
    ("hinihintay ko ang reply ng aking classmate", "neutral"),
    ("nasa second floor kami ng library ngayon", "neutral"),
    ("mag uuwi na kami pagkatapos ng klase", "neutral"),
    ("naka print na ang aming group output", "neutral"),
    ("the school announced a holiday next Monday", "neutral"),
    ("i need to review two more chapters tonight", "neutral"),
    ("the quiz will be held on Wednesday morning", "neutral"),
    ("please bring your ID card to the exam room", "neutral"),
    ("i am still waiting for my classmate to arrive", "neutral"),
    ("the professor posted the grades online already", "neutral"),
    ("i will read the feedback and revise my work", "neutral"),
    ("the campus is open from seven to six daily", "neutral"),
    ("i already enrolled in my subjects for next term", "neutral"),
    ("please check if the room is available tomorrow", "neutral"),
    ("naka submit na kami ng lahat ng requirements", "neutral"),
    ("bukas ang enrolment para sa susunod na sem", "neutral"),
    ("nakatanggap na kami ng schedule ng finals", "neutral"),
    ("nag email na si teacher ng study guide", "neutral"),
    ("magkikita kami ng grupo sa sabado", "neutral"),
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