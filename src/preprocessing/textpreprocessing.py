import re
import html
import logging
import string
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# ── NLTK data (download only if missing) ─────────────────────────────────────
for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Contractions map
# Expanded BEFORE tokenisation so negations survive as real words
# ─────────────────────────────────────────────────────────────────────────────
CONTRACTIONS: dict[str, str] = {
    "won't": "will not",
    "can't": "cannot",
    "don't": "do not",
    "didn't": "did not",
    "doesn't": "does not",
    "isn't": "is not",
    "wasn't": "was not",
    "aren't": "are not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "mustn't": "must not",
    "needn't": "need not",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "let's": "let us",
}

# Compiled once at import time for speed
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS) + r")\b",
    flags=re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PreprocessingConfig:
    """
    All knobs for the pipeline in one place.
    Use the class-methods for sensible domain defaults.
    """

    domain: str = "general"

    # ── cleaning flags ────────────────────────────────────────────────────────
    expand_contractions: bool = True   # "don't" → "do not" BEFORE tokenising
    strip_html: bool = False
    remove_urls: bool = True
    remove_mentions: bool = False      # strip @user
    keep_hashtag_text: bool = True     # #happy → happy  (True) or drop entirely
    remove_emojis: bool = True
    remove_numbers: bool = True
    lowercase: bool = True

    # ── token-level flags ─────────────────────────────────────────────────────
    remove_stopwords: bool = True
    keep_negations: bool = True        # always keep "not", "no", etc.
    use_stemming: bool = False
    use_lemmatization: bool = True
    min_token_len: int = 2

    # ── extra vocab ───────────────────────────────────────────────────────────
    extra_stopwords: list[str] = field(default_factory=list)
    protected_tokens: list[str] = field(default_factory=list)  # never removed

    # ── Named constructors ────────────────────────────────────────────────────
    @classmethod
    def for_reviews(cls) -> "PreprocessingConfig":
        """Amazon Fine Food Reviews — long structured text, may have HTML."""
        return cls(
            domain="food_reviews",
            strip_html=True,
            remove_mentions=False,
            keep_hashtag_text=True,
            min_token_len=2,
        )

    @classmethod
    def for_tweets(cls) -> "PreprocessingConfig":
        """Sentiment140 — short noisy social text."""
        return cls(
            domain="twitter",
            strip_html=False,
            remove_mentions=True,    # @user adds no sentiment signal
            keep_hashtag_text=True,  # #happy → happy (keep the word)
            min_token_len=3,         # filters more junk abbreviations
        )


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor
# ─────────────────────────────────────────────────────────────────────────────
_NEGATION_WORDS = frozenset({
    "not", "no", "nor", "never", "nothing", "nobody",
    "neither", "nowhere", "cannot",
})

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


class TextPreprocessor:
    """
    Configurable text cleaning + tokenisation pipeline.

    Parameters
    ----------
    config : PreprocessingConfig
        All preprocessing options.

    Examples
    --------
    >>> proc = TextPreprocessor(PreprocessingConfig.for_tweets())
    >>> proc.process("omg @starbucks this is LIT 🔥 http://t.co/abc")
    'omg lit'
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._stemmer = PorterStemmer()
        self._lemmatizer = WordNetLemmatizer()
        self._stopwords = self._build_stopwords()

    # ── public API ────────────────────────────────────────────────────────────

    def process(self, text: str) -> str:
        """Clean one string, return space-joined token string."""
        if not isinstance(text, str):
            if pd.isna(text):
                return ""
            text = str(text)

        text = self._clean(text)
        tokens = self._tokenize(text)
        tokens = self._filter_tokens(tokens)
        tokens = self._normalise_tokens(tokens)
        return " ".join(tokens)

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str,
        out_col: str = "clean_text",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Apply pipeline to a dataframe column in-place.

        Parameters
        ----------
        df : pd.DataFrame
        text_col : str   — source column name
        out_col : str    — destination column name (default: 'clean_text')
        show_progress : bool — show tqdm progress bar

        Returns
        -------
        pd.DataFrame  (same object, new column added)
        """
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in dataframe.")

        tqdm.pandas(desc=f"[{self.config.domain}] preprocessing")
        apply_fn = df[text_col].progress_apply if show_progress else df[text_col].apply
        df[out_col] = apply_fn(self.process)

        n_empty = (df[out_col] == "").sum()
        if n_empty:
            logger.warning("%d rows produced empty strings after preprocessing.", n_empty)

        return df

    def vocabulary(self, texts: list[str]) -> dict[str, int]:
        """Return token → count mapping across a list of texts."""
        from collections import Counter
        all_tokens = []
        for t in texts:
            all_tokens.extend(self.process(t).split())
        return dict(Counter(all_tokens).most_common())

    # ── private: cleaning ─────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        cfg = self.config

        # Unicode normalisation first — catches fancy quotes, ligatures, etc.
        text = unicodedata.normalize("NFKC", text)

        if cfg.lowercase:
            text = text.lower()

        # Contractions BEFORE tokenisation so "don't" → "do not" as full words
        if cfg.expand_contractions:
            text = _CONTRACTION_RE.sub(lambda m: CONTRACTIONS[m.group().lower()], text)

        if cfg.strip_html:
            text = html.unescape(text)
            text = re.sub(r"<[^>]+>", " ", text)

        if cfg.remove_urls:
            text = re.sub(r"http\S+|www\.\S+", " ", text)

        if cfg.remove_mentions:
            text = re.sub(r"@\w+", " ", text)

        # Hashtags: keep the word, drop the #
        if cfg.keep_hashtag_text:
            text = re.sub(r"#(\w+)", r"\1", text)
        else:
            text = re.sub(r"#\w+", " ", text)

        if cfg.remove_emojis:
            text = _EMOJI_RE.sub(" ", text)

        if cfg.remove_numbers:
            text = re.sub(r"\d+", " ", text)

        # Strip punctuation (after contractions are expanded)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ── private: tokenisation & filtering ────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        return word_tokenize(text)

    def _filter_tokens(self, tokens: list[str]) -> list[str]:
        cfg = self.config
        protected = set(cfg.protected_tokens)
        result = []

        for tok in tokens:
            # Always keep explicitly protected tokens
            if tok in protected:
                result.append(tok)
                continue

            # Length filter
            if len(tok) < cfg.min_token_len:
                continue

            # Stopword removal
            if cfg.remove_stopwords:
                is_negation = cfg.keep_negations and tok in _NEGATION_WORDS
                if not is_negation and tok in self._stopwords:
                    continue

            result.append(tok)

        return result

    def _normalise_tokens(self, tokens: list[str]) -> list[str]:
        if self.config.use_stemming:
            return [self._stemmer.stem(t) for t in tokens]
        if self.config.use_lemmatization:
            return [self._lemmatizer.lemmatize(t) for t in tokens]
        return tokens

    # ── private: helpers ──────────────────────────────────────────────────────

    def _build_stopwords(self) -> frozenset[str]:
        base = set(stopwords.words("english"))
        base.update(self.config.extra_stopwords)
        # Negations are handled separately in _filter_tokens
        # so we don't strip them from the set here — that's intentional.
        return frozenset(base)

    def __repr__(self) -> str:
        return f"TextPreprocessor(domain={self.config.domain!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test  (python preprocessing.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    review_proc = TextPreprocessor(PreprocessingConfig.for_reviews())
    tweet_proc  = TextPreprocessor(PreprocessingConfig.for_tweets())

    test_cases = [
        ("review", "This coffee is <b>amazing</b>!! Best I've ever tasted. 10/10."),
        ("review", "The product arrived damaged. I don't recommend this at all."),
        ("tweet",  "omg @starbucks this is LIT 🔥🔥 http://t.co/abc #blessed"),
        ("tweet",  "worst experience EVER, won't be buying again 😡 #terrible"),
        ("review", "No taste whatsoever. Not worth the money. Cannot believe I bought this."),
    ]

    header = f"{'TYPE':<8} {'RAW':<55} {'CLEANED'}"
    print(header)
    print("─" * 110)

    for dtype, raw in test_cases:
        proc    = review_proc if dtype == "review" else tweet_proc
        cleaned = proc.process(raw)
        print(f"{dtype:<8} {raw[:54]:<55} {cleaned}")

    print()

    # Negation preservation check
    neg_cases = [
        ("I don't like this product.", review_proc),
        ("won't be buying again", tweet_proc),
        ("Cannot believe how bad this is", review_proc),
    ]
    print("NEGATION PRESERVATION CHECK")
    print("─" * 60)
    for raw, proc in neg_cases:
        cleaned = proc.process(raw)
        preserved = any(neg in cleaned.split() for neg in _NEGATION_WORDS)
        status = "✓ kept" if preserved else "✗ LOST"
        print(f"  [{status}]  '{raw}'")
        print(f"           → '{cleaned}'")
        print()