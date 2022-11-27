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

"""
def xgb_learning(
    X: np.array,
    y: np.array,
    seed: int = 13,
    CV: int = 5,
    sample_weights: List[float] = None,
) -> List[xgboost.XGBClassifier]:

    xgbs = []

    split = split_by_class(y, CV=CV)

    for i, (train_idx, test_idx) in enumerate(split):

        X_train_i, X_test_i = X[train_idx], X[test_idx]
        y_train_i, y_test_i = y[train_idx], y[test_idx]

        # shuffle the elements
        seed = 13
        np.random.seed(seed)

        p = np.random.permutation(X_train_i.shape[0])
        X_train_i = X_train_i[p]
        y_train_i = y_train_i[p]

        sample_weights_i = None
        if sample_weights is not None:
            sample_weights_i = np.array(sample_weights)[train_idx]

        xgb = xgboost.XGBClassifier(
            objective="multi:softprob",
            n_estimators=100,
            eval_metric="auc",
            colsample_bytree=0.5,
            gamma=0,
            learning_rate=0.1,
            max_depth=3,
            reg_lambda=0,
            verbosity=1,
        )

        eval_set = [(dataset, targets)]  #
        xgb.fit(
            X_train_i,
            y_train_i,
            early_stopping_rounds=5,
            eval_set=eval_set,
            sample_weight=sample_weights_i,
            verbose=True,
        )

        y_pred = xgb.predict(X_test_i)
        kappa = quadratic_weighted_kappa(y_pred, y_test_i)
        print("Kappa = ", kappa)

        xgbs.append(xgb)

    return xgbs
"""


def xgb_learning(
    X: np.array,
    y: np.array,
    CV: int = 5,
    depth: int = 20,
    label_weights: List[float] = None,
    seed: int = 13,
    sample_weights: List[float] = None,
) -> List[xgboost.XGBClassifier]:
    label_weights = {
        0: label_weights[0],
        1: label_weights[1],
        2: label_weights[2],
    }

    xgbs = []

    kfold = KFold(n_splits=CV, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(kfold.split(X)):

        X_train_i, X_test_i = X[train_idx], X[test_idx]
        y_train_i, y_test_i = y[train_idx], y[test_idx]

        sample_weights_i = None
        if sample_weights is not None:
            sample_weights_i = np.array(sample_weights)[train_idx]

        xgb = xgboost.XGBClassifier(
            objective="multi:softprob",
            eval_metric="auc",
            colsample_bytree=0.5,
            gamma=0,
            learning_rate=0.1,
            max_depth=3,
            reg_lambda=0,
            verbosity=1,
        )

        eval_set = [(dataset, targets)]
        xgb.fit(
            X_train_i,
            y_train_i,
            early_stopping_rounds=5,
            eval_set=eval_set,
            sample_weight=sample_weights_i,
            verbose=True,
        )

        y_pred = xgb.predict(X_test_i)
        kappa = quadratic_weighted_kappa(y_pred, y_test_i)
        print("Kappa = ", kappa)

        xgbs.append(xgb)

    return xgbs


def predict_xgb_ensemble(xgbs: List[xgboost.XGBClassifier], X) -> np.array:
    predictions = []

    for xgb in xgbs:
        model_predictions = xgb.predict(X)
        predictions.append(model_predictions.reshape((-1, 1)))

    predictions = np.concatenate(predictions, axis=1)

    # count the number of class predictions for each sample
    num_predicted = [
        np.count_nonzero(predictions == i, axis=1).reshape((-1, 1))
        for i in range(3)
    ]
    num_predicted = np.concatenate(num_predicted, axis=1)

    # majority vote for final prediction
    final_predictions = np.argmax(num_predicted, axis=1)

    return final_predictions


if __name__ == "__main__":
    this_dir = os.path.dirname(os.getcwd())

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

    #    xgbs = xgb_learning(
    #        dataset,
    #        targets,
    #        CV=5,
    #        sample_weights=sample_weights,
    #        seed=seed,
    #   )

    xgbs = xgb_learning(
        dataset,
        targets,
        CV=5,
        label_weights=weights,
        sample_weights=sample_weights,
        seed=seed,
    )
    # load test set
    submission_ids, submission_smiles = load_test_data(test_path)
    # X = smiles_to_morgan_fp(submission_smiles)
    # descriptors
    qm_descriptors_test = smiles_to_qm_descriptors(
        submission_smiles, data_dir, "test"
    )
    # standardization of qm descriptors for the test set
    qm_descriptors_test = standardize_qm_test(
        qm_descriptors_test, columns_info
    )
    qm_descriptors_test = build_poly(qm_descriptors_test, columns_info, degree)

    # X = np.concatenate((qm_descriptors_test, X), axis=1)

    final_predictions = predict_xgb_ensemble(xgbs, qm_descriptors_test)

    submission_file = os.path.join(
        this_dir, "xg_boost_splitted_augmented_2.csv"
    )
    create_submission_file(submission_ids, final_predictions, submission_file)
