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

        # if result[i] == 2 and (
        #    x1[i, 1] != 2 or x2[i, 1] != 2 or x3[i, 1] != 2
        # ):
        #    if x1[i, 1] != 2:
        #        result[i] = x1[i, 1]
        #    elif x2[i, 1] != 2:
        #        result[i] = x2[i, 1]
        #    else:
        #        result[i] = x3[i, 1]

        if result[i] == 2 and (
            x1[i, 1] != 2 or x2[i, 1] != 2 or x3[i, 1] != 2
        ):
            if x1[i, 1] != 2:
                result[i] = x1[i, 1]
            elif x2[i, 1] == 1:  # Chemberta
                result[i] = 1
            elif x3[i, 1] == 0:  # Schnet
                result[i] = 0

        # if result[i] == 2 and (
        #    x1[i, 1] == 1 or x2[i, 1] == 1 or x3[i, 1] == 1
        # ):
        #    result[i] = 1

    return x1[:, 0], result


if __name__ == "__main__":

    this_dir = os.path.dirname(os.getcwd())

    # XGBOOST
    filename_xgboost = os.path.join(
        this_dir, "xg_boost_predictions_descriptors_weights_nonan.csv"
    )

    df1 = pd.read_csv(filename_xgboost)

    x_xgboost = df1[["Id", "Pred"]].values

    # CHEMBERTA
    filename_chemberta = os.path.join(this_dir, "Chemberta.csv")
    df2 = pd.read_csv(filename_chemberta)

    x_chem = df2[["Id", "Pred"]].values

    # SCHNET
    filename_schnet = os.path.join(this_dir, "SchNet.csv")
    df3 = pd.read_csv(filename_schnet)

    x_schnet = df3[["Id", "Pred"]].values

    ids, X = comparison(x_xgboost, x_chem, x_schnet)

    submission_file = os.path.join(this_dir, "comparison_17.csv")
    create_submission_file(ids, X, submission_file)
