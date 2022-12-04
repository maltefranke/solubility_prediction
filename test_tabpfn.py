import os
import csv
from typing import Tuple, List
import math
import torch as th
import numpy as np
from sklearn.linear_model import SGDClassifier

device = "cuda" if th.cuda.is_available() else "cpu"

from utils import *
from data_utils import *
from TabPFN_main.tabpfn.scripts.transformer_prediction_interface import (
    TabPFNClassifier,
)
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    this_dir = os.getcwd()

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # get data and transform smiles -> morgan fingerprint
    ids, smiles, targets = load_train_data(train_path)
    all_fps = smiles_to_morgan_fp(smiles)

    (
        all_fps_train,
        all_fps_test,
        targets_train,
        targets_test,
    ) = train_test_split(all_fps, targets, test_size=0.33, random_state=42)

    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=32)

    classifier.fit(all_fps_train[0:10], targets_train[0:10])
    targets_eval, p_eval = classifier.predict(
        all_fps_test[0:5], return_winning_probability=True
    )

    print("Accuracy", accuracy_score(targets_test[0:5], targets_eval))
