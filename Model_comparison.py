import pandas as pd
import os
import numpy as np
import statistics

from data_utils import *


def comparison(x1, x2, x3):

    result = np.zeros(x1.shape[0], dtype=int)

    for i in range(result.shape[0]):
        # if the 3 values are different, it takes the first one -> strongest method as the 1st
        result[i] = statistics.mode([x1[i, 1], x2[i, 1], x3[i, 1]])

    return x1[:, 0], result


if __name__ == "__main__":

    this_dir = os.path.dirname(os.getcwd())

    # XGBOOST
    filename_xgboost = os.path.join(this_dir, "submissions\xg_boost_predictions_descriptors_weights_nonan.csv")
    
    df1 = pd.read_csv(filename_xgboost)

    x_xgboost = df1[["Id", "Pred"]].values

    # CHEMBERTA
    filename_chemberta = os.path.join(
        this_dir, "submissions\Chemberta.csv"
    )
    df2 = pd.read_csv(filename_chemberta)

    x_chem = df2[["Id", "Pred"]].values

    # SCHNET
    filename_schnet = os.path.join(
        this_dir, "submissions\SchNet.csv"
    )
    df3 = pd.read_csv(filename_schnet)

    x_schnet = df3[["Id", "Pred"]].values

    ids, X = comparison(x_xgboost, x_chem, x_schnet)

    submission_file = os.path.join(this_dir, "comparison.csv")
    create_submission_file(ids, X, submission_file)
