import os
import pickle
from pathlib import Path

import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from zemberek import (
    TurkishMorphology,
    TurkishSentenceNormalizer,
    TurkishSpellChecker,
    TurkishTokenizer,
)

base_dir = Path(__file__).resolve().parent.parent
model_path = base_dir / "saved_models"
NOT_KERAS_MODELS = ["tokenizer", "encoder", "model"]

ABBREVIATIONS = {"mrb": "merhaba", "slm": "selam"}
LABELS = {"pos": "Positive", "neu": "Neutral", "neg": "Negative", "x": "Not mentioned"}
MAXLEN = 100

nltk.download("stopwords")
TURKISH_STOPWORDS = set(stopwords.words("turkish"))

MORPHOLOGY = TurkishMorphology.create_with_defaults()
NORMALIZER = TurkishSentenceNormalizer(MORPHOLOGY)
SPELLCHECKER = TurkishSpellChecker(MORPHOLOGY)
TOKENIZER = TurkishTokenizer.DEFAULT

SHOULD_BE_NORMALIZED = {
    "a 101": "a101",
    "daçathlon": "decathlon",
    "elin musk": "elon musk",
    "migors": "migros",
    "koçlaş": "koçtaş",
}

files = os.listdir(model_path)
MODELS = {}

for file in files:
    if file.endswith(".pkl"):
        name = file.split(".")[0]
        with open(model_path / file, "rb") as f:
            model = pickle.load(f)
    elif file.endswith(".keras"):
        name = " ".join(file.split("_")[:-1])
        model = tf.keras.models.load_model(model_path / file)

    MODELS[name] = model
