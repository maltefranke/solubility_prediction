import pandas as pd
import os
import numpy as np
import statistics
from typing import Tuple
from data_utils import load_train_data, load_test_data, create_submission_file


def comparison(x1:np.array, x2:np.array, x3:np.array)-> Tuple[np.array,np.array]:
    """
    Creates a .csv file with the predictions based on the predictions of three different models:
    XGBoost, ChemBERTa, SchNet by choosing, for each molecule, the minority class if predicted by one model, the most frequent class otherwise .

    Args:
        x1: XGBoost
        x2: ChemBERTa
        x3: SchNet

    Returns: ensemble_model

    """

    result = np.zeros(x1.shape[0], dtype=int)

    for i in range(result.shape[0]):
        # if the 3 values are different, it takes the first one -> strongest method as the 1st
        result[i] = statistics.mode([x1[i, 1], x2[i, 1], x3[i, 1]])

        if result[i] == 2 and (
            x1[i, 1] != 2 or x2[i, 1] != 2 or x3[i, 1] != 2
        ):
            if x1[i, 1] != 2:
                result[i] = x1[i, 1]
            elif x2[i, 1] != 2:
               result[i] = x2[i, 1]
            else:
               result[i] = x3[i, 1]

    return x1[:, 0], result


if __name__ == "__main__":

    this_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(this_dir, "../data")

    # XGBOOST
    filename_xgboost = os.path.join(
        data_dir, "XGBoost_best.csv"
    )

    df1 = pd.read_csv(filename_xgboost)

    x_xgboost = df1[["Id", "Pred"]].values

    # CHEMBERTA
    filename_chemberta = os.path.join(data_dir, "Chemberta.csv")
    df2 = pd.read_csv(filename_chemberta)

    x_chem = df2[["Id", "Pred"]].values

    # SCHNET
    filename_schnet = os.path.join(data_dir, "SchNet.csv")
    df3 = pd.read_csv(filename_schnet)

    x_schnet = df3[["Id", "Pred"]].values

    ids, X = comparison(x_xgboost, x_chem, x_schnet)

    submission_file = os.path.join(this_dir, "../submissions/ensemble_model.csv")
    create_submission_file(ids, X, submission_file)
