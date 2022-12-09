import os
import random

import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem import Draw

from data_utils import load_train_data
from rdkit import Chem as Chem

if __name__ == "__main__":

    this_dir = os.getcwd()

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # load smiles
    ids, smiles, targets = load_train_data(train_path)

    # Choose molecules to draw
    rnd_mol = [random.randint(0, len(ids)) for i in range(9)]
    selected_mol = [Chem.MolFromSmiles(smiles[i]) for i in rnd_mol]
    print(targets[rnd_mol])

    # Plot
    fig = Draw.MolsToGridImage(selected_mol, molsPerRow=3, subImgSize=(800, 800))
    fig.save('picture_molecule.png')
