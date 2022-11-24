import os
import csv
from typing import Tuple, List
import math
import numpy as np
from sklearn.linear_model import SGDClassifier

from utils import *
from data_preparation import *


def SVMlearning(
    X: np.array,
    y: np.array,
    CV: int = 5,
    depth: int = 20,
    label_weights: List[float] = None,
    seed: int = 13,
    sample_weights: List[float] = None,
) -> List[SGDClassifier]:
    # by default the SGDClassifier fits a linear support vector machine, with L2-norm penalization
    label_weights = {
        0: label_weights[0],
        1: label_weights[1],
        2: label_weights[2],
    }

    svms = []

    kfold = KFold(n_splits=CV, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(kfold.split(X)):

        X_train_i, X_test_i = X[train_idx], X[test_idx]
        y_train_i, y_test_i = y[train_idx], y[test_idx]

        sample_weights_i = None
        if sample_weights is not None:
            sample_weights_i = np.array(sample_weights)[train_idx]

        svm = SGDClassifier(class_weight=label_weights)

        svm.fit(X_train_i, y_train_i, sample_weight=sample_weights_i)

        y_pred = svm.predict(X_test_i)
        kappa = quadratic_weighted_kappa(y_pred, y_test_i)
        print("Kappa = ", kappa)

        svms.append(svm)

    return svms


def predict_svm_ensemble(svms: List[SGDClassifier], X) -> np.array:
    predictions = []

    for svm in svms:
        model_predictions = svm.predict(X)
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
    this_dir = os.getcwd()
    # this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # get data and transform smiles -> morgan fingerprint
    ids, smiles, targets = load_train_data(train_path)
    all_fps = smiles_to_morgan_fp(smiles)

    # GENERATE BALANCED CLASSES
    # Up/downsampling
    # targets, all_fps = up_down_sampling(targets, all_fps)

    # see how balanced the data is and assign weights
    train_data_size = targets.shape[0]

    num_low = np.count_nonzero(targets == 0)
    num_medium = np.count_nonzero(targets == 1)
    num_high = np.count_nonzero(targets == 2)

    weights = [
        1 - num_low / train_data_size,
        1 - num_medium / train_data_size,
        1 - num_high / train_data_size,
    ]
    print(weights)

    seed = 13
    np.random.seed(seed)

    # we permutate/shuffle our data first
    p = np.random.permutation(targets.shape[0])
    all_fps = all_fps[p]
    targets = targets[p]

    sample_weights = [weights[i] for i in targets]

    svms = SVMlearning(
        all_fps,
        targets,
        CV=5,
        label_weights=weights,
        sample_weights=sample_weights,
        seed=seed,
    )

    submission_ids, submission_smiles = load_test_data(test_path)
    X = smiles_to_morgan_fp(submission_smiles)
    input_dim = X.shape[-1]

    predictions = predict_svm_ensemble(svms, X)

    submission_file = os.path.join(this_dir, "predictions.csv")
    # create_submission_file(submission_ids, final_predictions, submission_file)