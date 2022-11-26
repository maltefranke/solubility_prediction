import os
import csv
from typing import Tuple, List
import math
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from utils import *
from data_preparation import *

"""
    param_grid = {
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.1, 0.01, 0.05],
        "gamma": [0, 0.25, 1],
        "max_delta_step": [0, 1, 5],
        "reg_lambda": [0, 1, 10],
        "alpha": [0, 1, 10],
    }
"""


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
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.1, 0.01, 0.05],
        "gamma": [0, 0.25, 1],
        "max_delta_step": [0, 1, 5],
        "reg_lambda": [0, 1, 10],
        "alpha": [0, 1, 10],
    }
    # Init classifier
    xgb_cl = xgboost.XGBClassifier(objective="multi:softprob")

    # Init Grid Search
    grid_cv = GridSearchCV(
        xgb_cl, param_grid, n_jobs=-1, cv=3, scoring="roc_auc"
    )

    # Fit
    _ = grid_cv.fit(X, y)

    grid_cv.best_score_

    print(grid_cv.best_params_)


if __name__ == "__main__":

    this_dir = os.path.dirname(os.getcwd())

    # data path
    data_dir = os.path.join(this_dir, "solubility_prediction\data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # get data and transform smiles -> morgan fingerprint
    ids, smiles, targets = load_train_data(train_path)

    # manipulation of the dataset
    degree = 2
    dataset, columns_info = preprocessing(ids, smiles, data_dir, degree)
    train_data_size = targets.shape[0]
    # we permute/shuffle our data first
    seed = 13
    np.random.seed(seed)
    p = np.random.permutation(targets.shape[0])
    dataset = dataset[p]
    targets = targets[p]

    weights = calculate_class_weights(targets)
    sample_weights = [weights[i] for i in targets]

    xgbs = xgb_cross(
        dataset,
        targets,
        label_weights=weights,
        sample_weights=sample_weights,
        seed=seed,
    )

"""
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(2, 5)
    for min_child_weight in range(1, 3)
]

cv_results = xgboost.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=13,
    nfold=3,
    metrics={"auc"},
    early_stopping_rounds=10,
)
cv_results

# ----------------------------------------------------------
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print(
        "CV with max_depth={}, min_child_weight={}".format(
            max_depth, min_child_weight
        )
    )
    # Update our parameters
    params["max_depth"] = max_depth
    params["min_child_weight"] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={"mae"},
        early_stopping_rounds=10,
    )
    # ------------------------------------
    # Update best MAE
    mean_mae = cv_results["test-mae-mean"].min()
    boost_rounds = cv_results["test-mae-mean"].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth, min_child_weight)
print(
    "Best params: {}, {}, MAE: {}".format(
        best_params[0], best_params[1], min_mae
    )
)
"""
