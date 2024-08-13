import logging
import pickle
from pathlib import Path

import nltk
from zemberek import (
    TurkishMorphology,
    TurkishSentenceNormalizer,
    TurkishSpellChecker,
    TurkishTokenizer,
)

logger = logging.getLogger(__name__)
# Sabitler
SHOULD_BE_NORMALIZED = {
    "a 101": "a101",
    "daçathlon": "decathlon",
    "elin musk": "elon musk",
    "migors": "migros",
    "koçlaş": "koçtaş",
}
ABBREVIATIONS = {"mrb": "merhaba", "slm": "selam"}
LABELS = {"pos": "Positive", "neu": "Neutral", "neg": "Negative", "x": "Not mentioned"}
MAXLEN = 128
MODEL_NAMES = [
    "yemek",
    "sirket",
    "motivasyon",
    "geri bildirim",
    "calisma ortami",
    "prim",
]

base_dir = Path(__file__).resolve().parent.parent
model_path = base_dir / "saved_models"

# Global variables
_initialized = False
MODELS = {}
TOKENIZER = None
ENCODER = None
TURKISH_STOPWORDS = None
MORPHOLOGY = None
NORMALIZER = None
SPELLCHECKER = None
ZEMBEREK_TOKENIZER = None


def initialize():
    global _initialized, MODELS, TOKENIZER, ENCODER, TURKISH_STOPWORDS, MORPHOLOGY, NORMALIZER, SPELLCHECKER, ZEMBEREK_TOKENIZER

    if _initialized:
        return

    nltk.download("stopwords")
    TURKISH_STOPWORDS = set(nltk.corpus.stopwords.words("turkish"))

    MORPHOLOGY = TurkishMorphology.create_with_defaults()
    NORMALIZER = TurkishSentenceNormalizer(MORPHOLOGY)
    SPELLCHECKER = TurkishSpellChecker(MORPHOLOGY)
    ZEMBEREK_TOKENIZER = TurkishTokenizer.DEFAULT

    with open(model_path / "encoder.pkl", "rb") as f:
        ENCODER = pickle.load(f)

    with open(model_path / "tokenizer.pkl", "rb") as f:
        TOKENIZER = pickle.load(f)

    logger.info("Models are initialized.")
    _initialized = True
