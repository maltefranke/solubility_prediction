import os
import site

import numpy as np
import torchmetrics

import schnetpack.src.schnetpack as spk
import schnetpack.src.schnetpack.transform as trn
import torch
from ase import Atoms
from schnetpack.src.schnetpack.data import ASEAtomsData
from schnetpack.src.schnetpack.data import AtomsDataModule

from data_preparation import *


def prepare_schnet_data(numbers: np.array, positions: np.array, targets: np.array, dataset: str="train"):
    # converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32)
    # data = converter(atoms)
    atoms_list = []
    for atom_numbers, positions in zip(numbers, positions):
        atoms = Atoms(numbers=atom_numbers, positions=positions)
        # atoms = Atoms(numbers=atom_numbers)
        atoms_list.append(atoms)

    targets = [{"solubility_class": int(i)} for i in targets]

    schnet_dataset = ASEAtomsData.create(
        f'./SchNet_{dataset}.db',
        distance_unit='Ang',
        property_unit_dict={'solubility_class': ''}
    )
    schnet_dataset.add_systems(targets, atoms_list)


def setup_schnet(dataset: str="train"):

    data = AtomsDataModule(
        f'SchNet_{dataset}.db',
        batch_size=8,
        num_train=0.8,
        num_val=0.1,
        num_test=0.1,
        property_units={'solubility_class': ''},
    )
    data.prepare_data()
    data.setup()

    cutoff = 5.
    n_atom_basis = 30

    pairwise_distance = spk.atomistic.PairwiseDistances()  # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )

    pred_solubility = spk.atomistic.Atomwise(n_in=n_atom_basis, n_out=3, output_key='solubility_class')

    nn_solubility = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_solubility],
    )

    output_solubility = spk.task.ModelOutput(
        name='solubility_class',
        loss_fn=torch.nn.CrossEntropyLoss(),
        loss_weight=1.,
        metrics={
            "Kappa": torchmetrics.CohenKappa(num_classes=3, weights="quadratic")
        }
    )

    task = spk.task.AtomisticTask(
        model=nn_solubility,
        outputs=[output_solubility],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4}
    )

    return task


if __name__ == "__main__":
    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # get data and transform smiles -> morgan fingerprint
    ids, smiles, targets = load_train_data(train_path)

    atomic_nums, positions = smiles_to_3d(smiles[:100])

    prepare_schnet_data(atomic_nums, positions, targets[:100])

    task = setup_schnet()

