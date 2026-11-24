#!/usr/bin/env python3
"""Hourly bike-rental demand forecasting (regression).

Two model families are compared with 5-fold cross-validation (RMSE) and the best one is saved:
  1. Ridge regression on one-hot categorical features + degree-2 polynomial features (alpha by CV)
  2. Gradient-boosted trees (HistGradientBoostingRegressor) with native categorical support
Rental demand has sharp hour x working-day interactions (commuter peaks) that trees
capture without hand-made interaction terms.
"""
import argparse
import lzma
import os
import pickle
import sys
import urllib.request
from typing import Optional

import numpy as np
import numpy.typing as npt
import sklearn.compose
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")

# Column order: season, year, month, hour, holiday, day of week, working day, weather,
#               temp, feeling temp, humidity, windspeed
CATEGORICAL = [0, 1, 2, 3, 4, 5, 7] #these are the index of the cathegorical variables, the other ones are numbers


class Dataset:
    """Rental dataset: 12 features per hour, target = number of rented bikes."""
    def __init__(self, name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def make_ridge() -> sklearn.pipeline.Pipeline:
    return sklearn.pipeline.Pipeline([
        ("prep", sklearn.compose.ColumnTransformer(
            [("cat", sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CATEGORICAL)],
            remainder="passthrough")),
        ("poly", sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)),
        ("scale", sklearn.preprocessing.StandardScaler()),
        ("ridge", sklearn.linear_model.RidgeCV(alphas=np.logspace(-2, 4, 20))), #linear regression with L2 penalization doing CV for the values of alpha
    ])


def make_trees(seed: int) -> sklearn.ensemble.HistGradientBoostingRegressor:
    return sklearn.ensemble.HistGradientBoostingRegressor(
        categorical_features=CATEGORICAL, learning_rate=0.08, max_iter=500,
        early_stopping=True, random_state=seed)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()
        candidates = {"ridge_poly2": make_ridge(), "hist_gradient_boosting": make_trees(args.seed)}
        cv = sklearn.model_selection.KFold(5, shuffle=True, random_state=args.seed)

        best_name, best_rmse = None, np.inf
        for name, candidate in candidates.items():
            scores = sklearn.model_selection.cross_val_score(
                candidate, train.data, train.target, cv=cv, scoring="neg_root_mean_squared_error")
            rmse = -scores.mean()
            print("{}: CV RMSE {:.2f} (+/- {:.2f})".format(name, rmse, scores.std()), file=sys.stderr)
            if rmse < best_rmse:
                best_name, best_rmse = name, rmse

        print("Selected model: {} (RMSE {:.2f})".format(best_name, best_rmse))
        model = candidates[best_name].fit(train.data, train.target)
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)
    else:
        test = Dataset(args.predict)
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        return np.clip(model.predict(test.data), 0, None)  # rentals cannot be negative


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)