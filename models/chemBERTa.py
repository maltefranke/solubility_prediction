import deepchem
from rdkit import Chem

# from apex import amp
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer
# from bertviz import head_view

import torch
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import rdFMCS
from matplotlib import colors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolToImage
from PIL import Image

import os

import numpy as np
import pandas as pd

from typing import List

# import molnet loaders from deepchem
from deepchem.molnet import load_bbbp, load_clearance, load_clintox, load_delaney, load_hiv, load_qm7, load_tox21

# import MolNet dataloder from bert-loves-chemistry fork
from molnet_dataloader import load_molnet_dataset, write_molnet_dataset_for_chemprop
import logging

from data_preparation import *

# tasks, (train_df, valid_df, test_df), transformers = load_molnet_dataset("clintox", tasks_wanted=None)

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import sklearn


def load_our_dataset(path: str, flag_test=False) -> Tuple[List[str], List[str], np.array]:

    df = pd.read_csv(path)
    del df['Id']  # delete id column

    if flag_test:
        df.rename(columns={'smiles': 'text'}, inplace=True)
        return df
    else:
        df.rename(columns={'smiles': 'text', 'sol_category': 'labels'}, inplace=True)
        df_train = df[0:int(0.8*df.shape[0])]
        df_valid = df[int(0.8*df.shape[0]):int(0.9*df.shape[0])]
        df_test = df[int(0.9*df.shape[0])+1:df.shape[0]]
        return df_train, df_valid, df_test


if __name__ == "__main__":
    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    ids, smiles, targets = load_train_data(train_path)

    weights = calculate_class_weights(targets)
    sample_weights = [weights[i] for i in targets]

    # set up a logger to record if any issues occur
    # and notify us if there are any problems with the arguments we've set for the model.
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    model = ClassificationModel('roberta', 'seyonec/PubChem10M_SMILES_BPE_396_250', num_labels=4,
                                # weight=sample_weights[0:int(0.8*train_data_size)],
                                args={'evaluate_each_epoch': True,
                                      'evaluate_during_training_verbose': True,
                                      'no_save': True, 'num_train_epochs': 10,
                                      'auto_weights': True}, use_cuda=False)


    train_df, valid_df, test_df = load_our_dataset(train_path, flag_test=False)

    train_data = [
        ["Aragorn was the heir of Isildur", 1],
        ["Frodo was the heir of Isildur", 0],
        ["Pippin is stronger than Merry", 2],
    ]
    train_df_for_checking = pd.DataFrame(train_data)
    train_df_for_checking.columns = ["text", "labels"]

    # Train the model
    model.train_model(train_df, eval_df=valid_df  #, output_dir='/content/BPE_PubChem_10M_ClinTox_run',
                      # args={'wandb_project': 'project-name'}
                      )

    # Test set
    submission_ids, submission_smiles = load_test_data(test_path)
    # X = smiles_to_morgan_fp(submission_smiles)

    submission_ids, submission_smiles = load_test_data(test_path)
    X = smiles_to_morgan_fp(submission_smiles)
    # descriptors
    # qm_descriptors_test = smiles_to_qm_descriptors(submission_smiles, data_dir, "test")

    # accuracy
    result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)

    # ROC-PRC
    result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.average_precision_score)

    final_predictions, raw_outputs = model.predict(test_df)

    submission_file = os.path.join(this_dir, "chemberta_submission1.csv")
    create_submission_file(submission_ids, final_predictions, submission_file)








