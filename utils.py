import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def quadratic_weighted_kappa(y_pred: np.array, y_true: np.array) -> float:
    """
    Function to compute the quadratic weighted kappa which is the primary metric for the competition
    Args:
        y_pred: np.array containing the model predictions
        y_true: np.array containing the true target values

    Returns:

    """
    # following https://www.kaggle.com/code/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps/notebook
    score = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return score



def differential_top_k_loss(y_pred, y_true):
    # following https://github.com/Felix-Petersen/difftopk
    pass
