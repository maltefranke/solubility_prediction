# Solubility classification of molecules

EPFL Machine Learning Course, Autumn 2022 - Class Project 2

Team members: Malte Franke, Laura Mismetti, Giacomo Mossinelli

This repository contains the code for the prediction of solubility class of compounds. In particular are distinguished compounds with:
- High solubility: Nephelometry values lower than 50000
- Medium solubility: Nephelometry values between 50000 and 100000
- Low solubility: Nephelometry values higher than 100000

# Overview
```data/``` - train and test set from Kaggle are placed here. Moreover, there are some subdivisions of the dataset<br> 
```models/``` - collection of models tested<br>
```submissions/``` - collection of some of the most relevant submissions<br>
```schnetpak```- copy of the schnetpack repository, modified to work for multi-class classification<br>
- ```ChemBERTa\``` - directory with ChemBERTa model code and results
- ```GraphModels\``` - directory with GNN results
- ```SchNet models\``` - directory for SchNet results and database
- ```ANN.py``` - artificial neural network workflow
- ```GraphModel.py``` - GNN model workflow
- ```SchNet.py``` - SchNet model workflow
- ```RandomForest.py``` - function which performs Random Forest multiclass Classification
- ```SVM.py``` - function which performs SVM multiclass Classification with sklearn.svm.SVC
- ```SVM_classification.py``` - function which performs SVM multiclass Classification with sklearn.linear_model.SGDClassifier
- ```XGboost.py``` - function which performs default XGBoost multiclass Classification

### Code
- ```model_comparison.py``` - function to create ensemble model starting from "submissions" files
- ```augmentation_utils.py``` - collection of the functions used to process and transform the data, as well as perform feature engineering
- ```conversion_smiles_utils.py``` - creation of molecular representations from SMILES 
- ```data_utils.py``` - functions to import dataset and deal with imbalance
- ```similarity_scaffolds.py``` 
- ```utils.py``` - Cohen's Kappa and Umap

# Environment
