import json
import re

import numpy as np
import requests
from django.http.response import JsonResponse
from tensorflow.keras.preprocessing.sequence import pad_sequences

from . import constants


def abbreviation_to_actual_word(text):
    tokens = constants.ZEMBEREK_TOKENIZER.tokenize(text)
    actual_words = [
        constants.ABBREVIATIONS.get(token.content, token.content) for token in tokens
    ]
    return " ".join(actual_words)


def split_titled_words(text):
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
    text = normalize(text)

    tokens = constants.ZEMBEREK_TOKENIZER.tokenize(text)
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


def decode_sentiment(y_pred):
    label = constants.ENCODER.inverse_transform(y_pred)[0]
    label = constants.LABELS.get(label, label)
    return label


def convert_text_to_sequence(x_test):
    sequence = pad_sequences(constants.TOKENIZER.texts_to_sequences(x_test), maxlen=300)
    return sequence.tolist()[0]


def predict(x_test):
    base_url = "http://tf-serving:8501/v1/models/"

    results = {"topics": {}}
    sequence = convert_text_to_sequence([x_test])

    for model_name in constants.MODEL_NAMES:
        # Model name
        model_url = "_".join(model_name.split()) + "_model:predict"
        payload = {"instances": [sequence]}

        response = requests.post(url=base_url + model_url, data=json.dumps(payload))
        response_data = response.json()
        predictions = np.array(response_data.get("predictions", []))

        overall_sentiment = decode_sentiment(np.argmax(predictions, axis=1))
        if overall_sentiment != "Not mentioned":
            results["topics"][model_name.title()] = {"emotions": {}}
            for idx, score in enumerate(predictions[0]):
                sentiment = decode_sentiment([idx])
                if sentiment == "Not mentioned":
                    continue
                results["topics"][model_name.title()]["emotions"].update(
                    {sentiment: str(score)}
                )

    return results
