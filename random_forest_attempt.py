import os
import csv
from typing import Tuple, List
import math
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem as Chem
from rdkit.Chem import AllChem as AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    log_loss,
    cohen_kappa_score,
)

from models.ANN import *
from data_preparation import *


if __name__ == "__main__":

    this_dir = os.getcwd()

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # get data and transform smiles -> morgan fingerprint
    ids, smiles, targets = load_train_data(train_path)
    all_fps = smiles_to_morgan_fp(smiles)

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

    index = math.floor(all_fps.shape[0] * 0.9)
    training_fps = all_fps[0:index]
    training_targets = targets[0:index]

    validation_fps = all_fps[index::]
    validation_targets = targets[index::]
    # -------------------------------------------------------------------------------------------
    # train an ensemble of ANNs, or load trained models if training was done previously
    # model_checkpoints = ann_learning(all_fps, targets, ann_save_path=os.path.join(this_dir, "TestResults"),
    #                                 label_weights=weights)
    # -------------------------------------------------------------------------------------------------
    # random forest on training set
    clf = RandomForestClassifier(random_state=0)
    clf.fit(training_fps, training_targets)

    # validation component
    validation_predictions = clf.predict(validation_fps)

    print(confusion_matrix(validation_targets, validation_predictions))
    print(classification_report(validation_targets, validation_predictions))
    print(accuracy_score(validation_targets, validation_predictions))

    kappa = cohen_kappa_score(validation_targets, validation_predictions)
    print("cohen kappa: ", kappa)
    # loss=log_loss(validation_targets,validation_predictions)
    # print("cross-entropy: ",loss)
    # --------------------------------------------------------------------------------------------
    submission_ids, submission_smiles = load_test_data(test_path)
    X = smiles_to_morgan_fp(submission_smiles)
    input_dim = X.shape[-1]

    # final_predictions = predict_ensemble(X, input_dim, model_checkpoints)
    final_predictions = clf.predict(X)

    submission_file = os.path.join(this_dir, "predictions.csv")
    create_submission_file(submission_ids, final_predictions, submission_file)
