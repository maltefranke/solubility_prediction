import os
import csv
from typing import Tuple, List
import math
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

import matplotlib.pyplot as plt
from augmentation_utils import *
from utils import *
from data_utils import *
from conversion_smiles_utils import *

"""
XGBoost cross validation
"""


if __name__ == "__main__":

    this_dir = os.path.dirname(os.getcwd())

    # data path
    data_dir = os.path.join(
        this_dir, "solubility_prediction/data"
    )  # CHANGE DEPENDING ON THE FOLDER!
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    print("importing train...")

    # load dataset
    ids, smiles, targets = load_train_data(train_path)

    # manipulation of the dataset
    print("PREPROCESSING...")
    dataset, columns_info, log_trans = preprocessing(ids, smiles, data_dir)
    train_data_size = targets.shape[0]

    # we permute/shuffle our data first
    seed = 13
    np.random.seed(seed)
    p = np.random.permutation(targets.shape[0])
    dataset = dataset[p]
    targets = targets[p]

    # TEST SET
    print("importing test...")

    submission_ids, submission_smiles = load_test_data(test_path)

    # TEST SET TRANSFORMATION
    # descriptors
    qm_descriptors_test = smiles_to_qm_descriptors(
        submission_smiles, data_dir, "test"
    )

    qm_descriptors_test, _ = transformation(
        qm_descriptors_test,
        submission_smiles,
        columns_info,
        standardization=False,
        test=True,
        degree=1,
        pairs=False,
        log_trans=log_trans,
        log=False,
        fps=True,
    )
    # features selection -> PCA

    # dataset, qm_descriptors_test = PCA_application(
    #    dataset, qm_descriptors_test
    # )

    # Computation of weights
    weights = calculate_class_weights(targets)
    sample_weights = [weights[i] for i in targets]
    label_weights = {
        0: weights[0],
        1: weights[1],
        2: weights[2],
    }
    # parameters collection
    param_grid = {
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "gamma": [0.0, 5.0, 10.0],
        "max_delta_step": [0, 5, 10],
        "reg_lambda": [0, 1, 5],
        "num_class": [3],
    }

    # Init classifier
    print("XGBoost...")
    xgb_cl = xgboost.XGBClassifier(
        objective="multi:softmax", tree_method="gpu_hist"
    )
    kappa_scorer = make_scorer(quadratic_weighted_kappa)

    grid_cv = GridSearchCV(xgb_cl, param_grid, cv=5, scoring=kappa_scorer)

    # Fit
    _ = grid_cv.fit(dataset, targets, sample_weight=sample_weights)
    print("results...")
    best_score = grid_cv.best_score_
    best_params = grid_cv.best_params_
    print(f"best squared kappa={grid_cv.best_score_}")
    print("BEST PARAMS")
    print(grid_cv.best_params_)
