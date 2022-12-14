import pandas as pd
import os
import numpy as np

from data_utils import *

"""
function to count the number of elements in a solubility class
"""


if __name__ == "__main__":

    this_dir = os.path.dirname(os.getcwd())

    filename = os.path.join(this_dir, "comparison_17.csv")

    df1 = pd.read_csv(filename)

    x = df1[["Id", "Pred"]].values
    value_1 = np.count_nonzero(x == 1)
    value_2 = np.count_nonzero(x == 2)
    value_0 = np.count_nonzero(x == 0)
    print(f"count 0: {value_0}\ncount 1: {value_1}\ncount 2: {value_2}")
