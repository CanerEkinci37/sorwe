import re

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from . import constants


def abbreviation_to_actual_word(text):
    """Converts abbreviations to their full forms in the text."""

    tokens = constants.TOKENIZER.tokenize(text)
    actual_words = [
        constants.ABBREVIATIONS.get(token.content, token.content) for token in tokens
    ]

    return " ".join(actual_words)


def split_titled_words(text):
    """Split words that are written in title case into individual words."""

    if text.isupper():
        return text

    char_list = []
    word_list = []
    for char in text:
        if char.islower() or char.isnumeric():
            char_list.append(char)
        else:
            if char_list:
                if char_list[-1].isupper():
                    char_list.append(char)
                else:
                    word_list.append("".join(char_list))
                    char_list = [char]
            else:
                char_list.append(char)

    word_list.append("".join(char_list))
    new_text = re.sub("[ ]+", " ", " ".join(word_list))

    return new_text


def normalize(text):
    """Normalizes the text by converting abbreviations, splitting titled words, and applying other normalization rules."""

    text = abbreviation_to_actual_word(text)
    text = split_titled_words(text)

    for key in constants.SHOULD_BE_NORMALIZED.keys():
        if key in text:
            text = text.replace(key, constants.SHOULD_BE_NORMALIZED[key])
        else:
            continue

    pattern = "^a-zA-ZçğıöşüÇĞİÖŞÜ"
    text = constants.NORMALIZER.normalize(text)

    if "a101" in text:
        text = re.sub(f"[{pattern}0-9 ]+", " ", text)
    else:
        text = re.sub(f"[{pattern} ]+", " ", text)
    text = re.sub("[ ]+", " ", text)

    return text.lower().strip()


def preprocess(text):
    """Preprocesses the text by normalizing and tokenizing it, and removing stopwords."""

    text = normalize(text)

    tokens = constants.TOKENIZER.tokenize(text)
    filtered_tokens = []

    for token in tokens:
        possible_words = []
        if token.content in constants.TURKISH_STOPWORDS:
            continue
        else:
            analyze = constants.MORPHOLOGY.analyze(token.content)
            for analysis in analyze:
                possible_words.append(analysis.item.normalized_lemma())
            if possible_words:
                filtered_tokens.append(possible_words[-1])
            else:
                filtered_tokens.append(token.content)

    return " ".join(filtered_tokens)


def to_sequence(text):
    """Converts text to a sequence of integers using the tokenizer."""

    tokenizer = constants.MODELS["tokenizer"]
    if isinstance(text, str):
        sequence = pad_sequences(
            tokenizer.texts_to_sequences([text]), maxlen=constants.MAXLEN
        )
    else:
        sequence = pad_sequences(
            tokenizer.texts_to_sequences(text), maxlen=constants.MAXLEN
        )
    return sequence


def decode_sentiment(y_pred):
    """Decodes the predicted sentiment label from model output."""

    label = constants.MODELS["encoder"].inverse_transform(y_pred)[0]
    label = constants.LABELS.get(label, label)

    return label


def count_sentiments(y_pred):
    """Return sentiment counts for dataset prediction."""
    sentiment_counts = {"Negative": 0, "Neutral": 0, "Positive": 0}

    for pred in y_pred:
        if pred != "Not mentioned":
            sentiment_counts[pred] += 1

    return sentiment_counts


def predict(x_test):
    """Predicts the sentiment labels for different aspects (yemek, şirket, motivasyon) based on the input text or dataset."""

    results = {"topics": {}}
    sequence = to_sequence(x_test)

    for name in constants.MODELS.keys():
        if name not in constants.NOT_KERAS_MODELS:
            y_pred = constants.MODELS[name].predict(sequence)

            if isinstance(x_test, str):
                confidence = round(np.max(y_pred), 3)
                sentiment = decode_sentiment(np.argmax(y_pred, axis=1))
                if sentiment == "Not mentioned":
                    continue
                results["topics"][name.title()] = {
                    "sentiment": sentiment,
                    "confidence": str(confidence),
                }
            else:
                y_pred = np.argmax(y_pred, axis=1)
                results["topics"][name.title()] = count_sentiments(
                    [decode_sentiment([pred]) for pred in y_pred]
                )

    return results
