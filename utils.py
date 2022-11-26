import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import umap

from data_preparation import *

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


def make_umap(X: np.array, y: np.array):
    class_indices = indices_by_class(y)
    class_0 = X[class_indices[0]]
    class_1 = X[class_indices[1]]
    class_2 = X[class_indices[2]]

    all_classes = np.concatenate([class_0, class_1, class_2], axis=0)

    # could append all fingerprints together -> make sure you know split them based on the label
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(all_classes)

    # plot all 3 labels
    plt.scatter(embedding[:len(class_0), 0], embedding[:len(class_0), 1], color='red', label="0")
    plt.scatter(embedding[len(class_0):len(class_0) + len(class_1), 0],
                embedding[len(class_0):len(class_0) + len(class_1), 1], color='blue', label="2")
    plt.scatter(embedding[len(class_0) + len(class_1):, 0], embedding[len(class_0) + len(class_1), 1],
                color='yellow', label="2")

    plt.show()



