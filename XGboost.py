import os
import csv
from typing import Tuple, List
import math
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score

import matplotlib.pyplot as plt
from utils import *
from data_preparation import *


if __name__ == "__main__":
    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(this_dir, "solubility_prediction\data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # get data and transform smiles -> morgan fingerprint
    ids, smiles, targets = load_train_data(train_path)
    all_fps = smiles_to_morgan_fp(smiles)
    # introduce descriptores
    qm_descriptors = smiles_to_qm_descriptors(smiles, data_dir)
    all_fps = np.concatenate((qm_descriptors, all_fps), axis=1)

    # we permute/shuffle our data first
    seed = 13
    np.random.seed(seed)

    p = np.random.permutation(targets.shape[0])
    all_fps = all_fps[p]
    targets = targets[p]

    xgb = xgboost.XGBClassifier()
    model_xgboost = xgboost.XGBClassifier(
        learning_rate=0.1,
        max_depth=5,
        n_estimators=500,
        subsample=0.5,
        colsample_bytree=0.5,
        eval_metric="auc",
        verbosity=1,
    )

    eval_set = [(all_fps, targets)]
    model_xgboost.fit(
        all_fps,
        targets,
        early_stopping_rounds=10,
        eval_set=eval_set,
        verbose=True,
    )

    submission_ids, submission_smiles = load_test_data(test_path)
    X = smiles_to_morgan_fp(submission_smiles)
    input_dim = X.shape[-1]

    predictions_train = model_xgboost.predict(all_fps)
    # final_predictions = model_xgboost.predict(X)

    kappa = sklearn.metrics.cohen_kappa_score(targets, predictions_train)
    print("Cohen's Kappa Train: {:.4f}".format(kappa))

    # submission_file = os.path.join(this_dir, "xg_boost_predictions_descriptors.csv")
    # create_submission_file(submission_ids, final_predictions, submission_file)
