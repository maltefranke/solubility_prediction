import os
import csv
from typing import Tuple, List
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from augmentation_utils import *
from utils import *
from data_utils import *
from conversion_smiles_utils import *


def rf_learning(
    X: np.array,
    y: np.array,
    CV: int = 5,
    label_weights: List[float] = None,
    seed: int = 13,
    sample_weights: List[float] = None,
) -> List[RandomForestClassifier]:

    """
    creation of the Random Forest model testing 5 splitting of the dataset
    """
    label_weights = {
        0: label_weights[0],
        1: label_weights[1],
        2: label_weights[2],
    }

    rfs = []

    kfold = KFold(n_splits=CV, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(kfold.split(X)):

        X_train_i, X_test_i = X[train_idx], X[test_idx]
        y_train_i, y_test_i = y[train_idx], y[test_idx]

        sample_weights_i = None
        if sample_weights is not None:
            sample_weights_i = np.array(sample_weights)[train_idx]

        rf = RandomForestClassifier(
            n_estimators=110,
            max_depth=20,
            criterion="entropy",
            random_state=seed,
            class_weight=label_weights,
        )

        rf.fit(X_train_i, y_train_i, sample_weight=sample_weights_i)

        y_pred = rf.predict(X_test_i)
        kappa = quadratic_weighted_kappa(y_pred, y_test_i)
        print("Kappa = ", kappa)

        rfs.append(rf)

    return rfs


def predict_rf_ensemble(rfs: List[RandomForestClassifier], X) -> np.array:
    """
    creation of the output with using previously built models
    """
    predictions = []

    for rf in rfs:
        model_predictions = rf.predict(X)
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

    data_dir = os.path.join(
        this_dir, "data"
    )  # MODIFY depending on your folder!!
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # get data and transform smiles
    print("loading data...\n")
    ids, smiles, targets = load_train_data(train_path)

    # DATASET TRANSFORMATION
    start = time.time()
    print("transformation...\n")
    dataset, columns_info, log_trans = preprocessing(
        ids,
        smiles,
        data_dir,
        nan_tolerance=0.0,
        standardization=True,
        cat_del=True,
        log=True,
        fps=False,
        degree=1,
        pairs=False,
    )

    # we permute/shuffle our data first
    seed = 13
    np.random.seed(seed)
    p = np.random.permutation(targets.shape[0])
    dataset = dataset[p]
    targets = targets[p]

    # TEST SET
    print("loading test set...\n")
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
        standardization=True,
        test=True,
        degree=1,
        pairs=False,
        log_trans=log_trans,
        log=True,
        fps=False,
    )

    # application of the PCA
    # print("PCA...\n")
    # dataset, qm_descriptors_test = PCA_application(
    #    dataset, qm_descriptors_test
    # )

    weights = calculate_class_weights(targets)
    sample_weights = [weights[i] for i in targets]

    print("Random Forest...\n")
    rfs = rf_learning(
        dataset,
        targets,
        CV=5,
        label_weights=weights,
        sample_weights=sample_weights,
        seed=seed,
    )

    final_predictions = predict_rf_ensemble(rfs, qm_descriptors_test)

    submission_file = os.path.join(this_dir, "submission.csv")
    create_submission_file(submission_ids, final_predictions, submission_file)
