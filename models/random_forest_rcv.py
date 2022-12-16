import os
import csv
from typing import Tuple, List
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from augmentation_utils import *
from utils import *
from data_utils import *
from conversion_smiles_utils import *
from sklearn.metrics import make_scorer

# {'n_estimators': 110, 'max_features': 'sqrt', 'max_depth': 5}
# random_forest_post_cv_0.00572
def rf_cross_validation(
    X: np.array,
    y: np.array,
    CV: int = 5,
    label_weights: List[float] = None,
    seed: int = 13,
    sample_weights: List[float] = None,
):
    label_weights = {
        0: label_weights[0],
        1: label_weights[1],
        2: label_weights[2],
    }

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=20, stop=200, num=5)]
    # Number of features to consider at every split
    max_features = ["None", "sqrt"]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 50, num=3)]
    forest = RandomForestClassifier(criterion="entropy", n_jobs=-1)
    # Create the random grid
    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
    }
    kappa_scorer = make_scorer(quadratic_weighted_kappa)
    rf_random = RandomizedSearchCV(
        estimator=forest,
        param_distributions=random_grid,
        n_iter=10,
        cv=5,
        verbose=2,
        random_state=seed,
        n_jobs=-1,
        scoring=kappa_scorer,
    )
    # Fit the random search model
    rf_random.fit(X, y, sample_weight=sample_weights)

    print("results...")
    print(rf_random.best_score_)
    print(rf_random.cv_results_)
    print("---")
    print(rf_random.best_params_)
    print("---")


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

    weights = calculate_class_weights(targets)
    sample_weights = [weights[i] for i in targets]

    print("Randomized CV Random Forest...\n")
    rfs = rf_cross_validation(
        dataset,
        targets,
        CV=5,
        label_weights=weights,
        sample_weights=sample_weights,
        seed=seed,
    )
