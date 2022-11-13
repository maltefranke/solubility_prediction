import os
import csv
from typing import Tuple, List
import math
import numpy as np
from sklearn.svm import SVC, OneClassSVM

from utils import *
from data_preparation import *


def composite_prediction(X, oc_svm: OneClassSVM, svm: SVC):

    # first, the oc svm predicts if a data point is class 2
    composite_predictions = oc_svm.predict(X)
    composite_predictions = np.where(composite_predictions != 1, composite_predictions, 2)

    # then let the svm predict if its class 0 or 1
    not_class2_idx = np.argwhere(composite_predictions != 2).squeeze()

    X_not_class2 = X[not_class2_idx]

    svm_predictions = svm.predict(X_not_class2)

    composite_predictions[not_class2_idx] = svm_predictions

    return composite_predictions


def composite_svm_learning(X: np.array, y: np.array, CV: int = 5, seed: int = 13) \
        -> Tuple[List[OneClassSVM], List[SVC]]:

    oc_svms = []
    svms = []

    kfold = KFold(n_splits=CV, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(kfold.split(X)):

        X_train_i, X_test_i = X[train_idx], X[test_idx]
        y_train_i, y_test_i = y[train_idx], y[test_idx]

        # Step 1: extract all datapoints with label 2 as this is the most prevalent class
        class2_idx = np.argwhere(y_train_i == 2).squeeze()
        X_class2 = X_train_i[class2_idx]

        # Step 2: train one-class SVM on this data
        oc_svm = OneClassSVM()
        oc_svm.fit(X_class2)

        oc_svms.append(oc_svm)

        X_class01 = np.delete(X_train_i, class2_idx)
        y_class01 = np.delete(y_train_i, class2_idx)

        svm = SVC(random_state=seed)

        svm.fit(X_class01, y_class01)

        y_pred = composite_prediction(X_train_i, oc_svm, svm)

        kappa = quadratic_weighted_kappa(y_pred, y_test_i)
        print("Kappa = ", kappa)

        svms.append(svm)

    return oc_svms, svms


def predict_composite_svm_ensemble(oc_svms: List[OneClassSVM], svms: List[SVC], X) -> np.array:
    predictions = []

    for oc_svm, svm in zip(oc_svms, svms):
        model_predictions = composite_prediction(X, oc_svm, svm)
        predictions.append(model_predictions.reshape((-1, 1)))

    predictions = np.concatenate(predictions, axis=1)

    # count the number of class predictions for each sample
    num_predicted = [np.count_nonzero(predictions == i, axis=1).reshape((-1, 1)) for i in range(3)]
    num_predicted = np.concatenate(num_predicted, axis=1)

    # majority vote for final prediction
    final_predictions = np.argmax(num_predicted, axis=1)

    return final_predictions


if __name__ == "__main__":

    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # get data and transform smiles -> morgan fingerprint
    ids, smiles, targets = load_train_data(train_path)
    all_fps = smiles_to_morgan_fp(smiles)

    """class0_idx = np.argwhere(targets == 0)
    class1_idx = np.argwhere(targets == 1)
    class2_idx = np.argwhere(targets == 2)

    class2_idx = class2_idx[:class1_idx.shape[0]]

    all_idx = np.concatenate([class0_idx, class1_idx, class2_idx])

    all_fps = all_fps[all_idx, :].squeeze()
    targets = targets[all_idx].squeeze()"""

    seed = 13
    np.random.seed(seed)

    # we permutate/shuffle our data first
    p = np.random.permutation(targets.shape[0])
    all_fps = all_fps[p]
    targets = targets[p]

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

    sample_weights = [weights[i] for i in targets]

    oc_svms, svms = composite_svm_learning(all_fps, targets, CV=5, seed=seed)

    submission_ids, submission_smiles = load_test_data(test_path)
    X = smiles_to_morgan_fp(submission_smiles)
    input_dim = X.shape[-1]

    predictions = predict_composite_svm_ensemble(oc_svms, svms, X)

    submission_file = os.path.join(this_dir, "predictions.csv")
    create_submission_file(submission_ids, predictions, submission_file)
