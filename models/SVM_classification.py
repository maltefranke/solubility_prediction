import os
import csv
from typing import Tuple, List
import math
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import time
from augmentation_utils import *
from utils import *
from data_utils import *
from conversion_smiles_utils import *


def SVMlearning(
    X: np.array,
    y: np.array,
    CV: int = 5,
    label_weights: List[float] = None,
    seed: int = 13,
    sample_weights: List[float] = None,
) -> List[SGDClassifier]:

    """
    creation of the SVM model testing 5 splitting of the dataset
    """
    # weights per class
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

        svm = SGDClassifier(
            penalty="l1",
            learning_rate="constant",
            eta0=0.01,
            class_weight=label_weights,
        )

        svm.fit(X_train_i, y_train_i, sample_weight=sample_weights_i)

        y_pred = svm.predict(X_test_i)
        kappa = quadratic_weighted_kappa(y_pred, y_test_i)
        print("Kappa = ", kappa)

        svms.append(svm)

    return svms


def predict_svm_ensemble(svms: List[SGDClassifier], X) -> np.array:
    """
    creation of the output with using previously built models
    """
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


def svm_cross_validation(
    X: np.array,
    y: np.array,
    CV: int = 5,
    label_weights: List[float] = None,
    seed: int = 13,
    sample_weights: List[float] = None,
):
    print("SVM random_CV...'\n'")
    label_weights = {
        0: label_weights[0],
        1: label_weights[1],
        2: label_weights[2],
    }

    learning_rate = ["constant", "optimal", "invscaling"]
    eta0 = [0.1, 0.05, 0.01]
    penalty = ["l1", "l2"]
    # Create the random grid
    random_grid = {
        "learning_rate": learning_rate,
        "eta0": eta0,
        "penalty": penalty,
    }

    clf = SGDClassifier(class_weight=label_weights)

    kappa_scorer = make_scorer(quadratic_weighted_kappa)
    gs = RandomizedSearchCV(
        clf,
        param_distributions=random_grid,
        n_iter=10,
        cv=5,
        verbose=2,
        random_state=seed,
        n_jobs=-1,
        scoring=kappa_scorer,
    )
    # -----------------------------------------------------
    # Train model
    gs.fit(X, y, sample_weight=sample_weights)

    print("results...")
    print(gs.best_score_)
    print(gs.cv_results_)
    print("---")
    print(gs.best_params_)
    print("---")


if __name__ == "__main__":
    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(
        this_dir, "solubility_prediction\data"
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

    r_cv = False
    if r_cv:
        svm_cross_validation(
            dataset,
            targets,
            CV=5,
            label_weights=weights,
            sample_weights=sample_weights,
            seed=seed,
        )

    print("SVM...\n")
    svms = SVMlearning(
        dataset,
        targets,
        CV=5,
        label_weights=weights,
        sample_weights=sample_weights,
        seed=seed,
    )

    final_predictions = predict_svm_ensemble(svms, qm_descriptors_test)

    submission_file = os.path.join(this_dir, "SVM_fps.csv")
    create_submission_file(submission_ids, final_predictions, submission_file)
