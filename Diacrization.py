#!/usr/bin/env python3
"""Czech diacritics restoration (e.g. "Dobry den, jak se mate?" -> "Dobrý den, jak se máte?").

Every letter that *could* carry a diacritic (a c d e i n o r s t u y z) is a
classification problem: "which of its possible forms is it?".  One multinomial
logistic regression is trained per base letter, so the model can never output
an impossible form (e.g. an 'a' turning into 'š').

Features for each position:
  * letters: the characters at offsets -window..+window (default 5 each side)
  * n-grams: every character n-gram (2..max_ngram, default 5) that contains the position
  * marks:   uppercase flag, word-start / word-end flags and the type of the
             neighbouring characters (letter / digit / space / punctuation)
Features are hashed (FeatureHasher), so memory stays bounded.
"""
import argparse
import lzma
import os
import pickle
import sys
import urllib.request
from typing import Optional

import numpy as np
import sklearn.feature_extraction
import sklearn.linear_model

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--window", default=5, type=int, help="Letters of context on each side")
parser.add_argument("--max_ngram", default=5, type=int, help="Longest character n-gram")
parser.add_argument("--C", default=1.0, type=float, help="Inverse L2 regularisation of the logistic regression")
parser.add_argument("--hash_bits", default=18, type=int, help="log2 of the number of hashed features")
parser.add_argument("--dev_fraction", default=0.05, type=float, help="Tail of the train text used for validation")


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self, name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2627/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)


NODIA_SET = set(Dataset.LETTERS_NODIA)
PAD = "\0"


def safe_lower(text: str) -> str:
    """Lowercase while guaranteeing the string keeps its exact length."""
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)


def char_class(c: str) -> str:
    if c.isalpha():
        return "L"
    if c.isdigit():
        return "D"
    if c.isspace():
        return "S"
    return "P"

#returns the index of the diacrizable letters of the text. (sombe characters can't be diacrizated)
def diacritizable_positions(low: str) -> np.ndarray:
    return np.array([i for i, c in enumerate(low) if c in NODIA_SET], dtype=np.int64)


def extract_features(data: str, low: str, positions, window: int, max_ngram: int):
    """Yield a list of string features for every requested position."""
    pad = max(window, max_ngram - 1) + 1
    padded_low = PAD * pad + low + PAD * pad
    padded_orig = PAD * pad + data + PAD * pad
    for i in positions:
        j = i + pad #because we've added a PAD to the text
        feats = []
        # 1) letters in the window
        for off in range(-window, window + 1):
            if off:
                feats.append("c{}={}".format(off, padded_low[j + off]))
        # 2) character n-grams that contain the current position
        for n in range(2, max_ngram + 1):
            for start in range(-(n - 1), 1):
                feats.append("n{}@{}={}".format(n, start, padded_low[j + start:j + start + n]))
        # 3) marks: case, word boundaries, neighbour character types
        feats.append("upper" if padded_orig[j].isupper() else "lower")
        feats.append("prev_upper" if padded_orig[j - 1].isupper() else "prev_notupper")
        feats.append("word_start" if not padded_low[j - 1].isalpha() else "in_word_left")
        feats.append("word_end" if not padded_low[j + 1].isalpha() else "in_word_right")
        feats.append("nb={}{}".format(char_class(padded_low[j - 1]), char_class(padded_low[j + 1])))
        yield feats


def make_hasher(hash_bits: int) -> sklearn.feature_extraction.FeatureHasher:
    return sklearn.feature_extraction.FeatureHasher(
        n_features=2 ** hash_bits, input_type="string", alternate_sign=False, dtype=np.float32)


def predict_text(models: dict, hasher, text: str, window: int, max_ngram: int) -> str:
    low = safe_lower(text)
    positions = diacritizable_positions(low)
    result = list(text)
    if not len(positions):
        return text
    #X is a matrix of number of rows the number of diacrizable letters we have analyzed and 262.144 columns
    X = hasher.transform(extract_features(text, low, positions, window, max_ngram)) 
    letters = np.array([low[i] for i in positions]) #creates a list with the diacrizable letters
    for letter, clf in models.items():
        rows = np.flatnonzero(letters == letter)
        if not len(rows):
            continue
        preds = [clf] * len(rows) if isinstance(clf, str) else clf.predict(X[rows]) #if its a string it only has one option and we saved that option as a string
        for r, p in zip(rows, preds):
            i = positions[r]
            result[i] = p.upper() if text[i].isupper() else p
    return "".join(result)


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()
        data, low, target = train.data, safe_lower(train.data), safe_lower(train.target)

        positions = diacritizable_positions(low)
        n_dev = int(len(positions) * args.dev_fraction) #the number of words we are saving for validation
        train_pos, dev_pos = positions[:len(positions) - n_dev], positions[len(positions) - n_dev:]

        hasher = make_hasher(args.hash_bits)
        print("Extracting features for {} positions.".format(len(train_pos)), file=sys.stderr)
        X = hasher.transform(extract_features(data, low, train_pos, args.window, args.max_ngram))
        letters = np.array([low[i] for i in train_pos])
        labels = np.array([target[i] for i in train_pos])

        models = {}
        for letter in sorted(NODIA_SET):
            rows = np.flatnonzero(letters == letter) #returns the indexof the positions that are True
            if not len(rows):
                continue
            classes = np.unique(labels[rows])
            if len(classes) == 1:           # e.g. a letter that never carries a diacritic in the data
                models[letter] = str(classes[0])
                continue
            clf = sklearn.linear_model.LogisticRegression(C=args.C, max_iter=200)
            clf.fit(X[rows], labels[rows])
            models[letter] = clf
            print("  trained '{}' ({} samples, classes {})".format(letter, len(rows), "".join(clf.classes_)),
                  file=sys.stderr)

        if n_dev:
            dev_text = data[dev_pos[0]:]
            dev_target = target[dev_pos[0]:]
            predicted = safe_lower(predict_text(models, hasher, dev_text, args.window, args.max_ngram))
            dev_low = safe_lower(dev_text)
            idx = diacritizable_positions(dev_low)
            accuracy = np.mean([predicted[i] == dev_target[i] for i in idx])
            print("Validation accuracy on diacritizable letters: {:.2f}%".format(100 * accuracy))

        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump({"models": models, "args": {"window": args.window, "max_ngram": args.max_ngram,
                                                    "hash_bits": args.hash_bits}}, model_file)
    else:
        test = Dataset(args.predict)
        with lzma.open(args.model_path, "rb") as model_file:
            saved = pickle.load(model_file)
        cfg = saved["args"]
        return predict_text(saved["models"], make_hasher(cfg["hash_bits"]), test.data,
                            cfg["window"], cfg["max_ngram"])


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)