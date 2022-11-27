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
        ind_0 = np.array(df[df['labels'] == 0].index)
        ind_1 = np.array(df[df['labels'] == 1].index)
        ind_2 = np.array(df[df['labels'] == 2].index)
        print(ind_0)
        ind_train = np.hstack([ind_0[0:int(0.9*ind_0.shape[0])],
                          ind_1[0:int(0.9*ind_1.shape[0])],
                          ind_2[0:int(0.9*ind_2.shape[0])]])
        ind_valid = np.hstack([ind_0[int(0.9*ind_0.shape[0]):ind_0.shape[0]],
                          ind_1[int(0.9*ind_1.shape[0]):ind_1.shape[0]],
                          ind_2[int(0.9*ind_2.shape[0]):ind_2.shape[0]]])
        df_train = df.iloc[ind_train]
        df_valid = df.iloc[ind_valid]

        print(df_train.head())
        print(df_valid.head())
        return df_train, df_valid


if __name__ == "__main__":
    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # set up a logger to record if any issues occur
    # and notify us if there are any problems with the arguments we've set for the model.
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    model = ClassificationModel('roberta', 'seyonec/PubChem10M_SMILES_BPE_396_250', num_labels=3,
                                # weight=sample_weights[0:int(0.8*train_data_size)],
                                args={'evaluate_each_epoch': True,
                                      'evaluate_during_training_verbose': True,
                                      'no_save': True, 'num_train_epochs': 10,
                                      'auto_weights': True}, use_cuda=False)

    train_df, valid_df = load_our_dataset(train_path, flag_test=False)

    model.train_model(train_df, eval_df=valid_df, output_dir='outputs', multi_label=True, num_labels=3, use_cuda=False,
                      args={'wandb_project': 'project-name'})


    # Test set
    test_df = load_our_dataset(test_path, flag_test=True)
    submission_ids, submission_smiles = load_test_data(test_path)

    # accuracy
    result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)

    # ROC-PRC
    result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.average_precision_score)

    final_predictions, raw_outputs = model.predict(test_df)

    submission_file = os.path.join(this_dir, "chemberta_submission1.csv")
    create_submission_file(submission_ids, final_predictions, submission_file)








