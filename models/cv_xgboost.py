import os
import csv
from typing import Tuple, List
import math
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import matplotlib.pyplot as plt
from utils import *
from data_preparation import *


def xgb_cross(
    X: np.array,
    y: np.array,
    label_weights: List[float] = None,
    seed: int = 13,
    sample_weights: List[float] = None,
):

    np.random.seed(seed)
    p = np.random.permutation(targets.shape[0])
    X = X[p]
    y = y[p]

    param_grid = {
        "max_depth": [2, 3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.25],
        "gamma": [0.0, 0.2, 0, 4, 0.5, 0.7, 0.9, 1],
        "max_delta_step": [0, 1, 3, 5, 7, 10],
        "reg_lambda": [0, 1, 5, 10],
        "alpha": [0, 1, 5, 10],
        "num_class": [3],
    }
    # Init classifier
    xgb_cl = xgboost.XGBClassifier(objective="multi:softmax")
    kappa_scorer = make_scorer(cohen_kappa_score)
    # Init Grid Search
    grid_cv = GridSearchCV(
        xgb_cl, param_grid, n_jobs=-1, cv=5, scoring=kappa_scorer
    )

    # Fit
    _ = grid_cv.fit(X, y)

    grid_cv.best_score_

    print(grid_cv.best_params_)


if __name__ == "__main__":

    this_dir = os.path.dirname(os.getcwd())

    # data path
    data_dir = os.path.join(this_dir, "solubility_prediction/data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # get data and transform smiles -> morgan fingerprint
    ids, smiles, targets = load_train_data(train_path)

    # manipulation of the dataset
    # degree = 2
    dataset, columns_info = preprocessing(ids, smiles, data_dir)
    train_data_size = targets.shape[0]
    # we permute/shuffle our data first
    seed = 13
    np.random.seed(seed)
    p = np.random.permutation(targets.shape[0])
    dataset = dataset[p]
    targets = targets[p]

    weights = calculate_class_weights(targets)
    print(type(weights))
    sample_weights = [weights[i] for i in targets]
    label_weights = {
        0: weights[0],
        1: weights[1],
        2: weights[2],
    }

    """
    xgbs = xgb_cross(
        dataset,
        targets,
        label_weights=weights,
        sample_weights=sample_weights,
        seed=seed,
    )
    """
    param_grid = {
        "max_depth": [2, 3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.25],
        "gamma": [0.0, 0.2, 0, 4, 0.5, 0.7, 0.9, 1],
        "max_delta_step": [0, 1, 3, 5, 7, 10],
        "reg_lambda": [0, 1, 5, 10],
        "alpha": [0, 1, 5, 10],
        "num_class": [3],
    }
    # Init classifier
    xgb_cl = xgboost.XGBClassifier(objective="multi:softmax")
    kappa_scorer = make_scorer(cohen_kappa_score)
    # Init Grid Search
    # https://stackoverflow.com/questions/13051706/using-sample-weight-in-gridsearchcv

    grid_cv = GridSearchCV(xgb_cl, param_grid, cv=5, scoring=kappa_scorer)

    # Fit
    _ = grid_cv.fit(dataset, targets, fit_params={"sample_weight": weights})

    print(grid_cv.best_score_)

    print(grid_cv.best_params_)
