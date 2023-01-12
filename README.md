# Solubility classification of molecules

EPFL Machine Learning Course, Autumn 2022 - Class Project 2

Team members: Malte Franke, Laura Mismetti, Giacomo Mossinelli

This repository contains the code for the prediction of solubility class of compounds. In particular, compounds are distinguished by the following:
- High solubility: Nephelometry values lower than 50000
- Medium solubility: Nephelometry values between 50000 and 100000
- Low solubility: Nephelometry values higher than 100000

The data included in this repository was taken from the Kaggle challenge [1st EUOS/SLAS Joint Challenge: Compound Solubility](https://www.kaggle.com/competitions/euos-slas/overview).
Some of the approaches used in this work require the transformation from SMILES strings to another machine-readable representation.
While all of the representations can be produced with our code, translating to molecular descriptors and Schnet-databases takes a long time and was only done once. 
These files are too large for git, and we encourage the users to contact us if interested, so that we can share the pre-computed datasets.

# Overview
```data/``` - train and test set from Kaggle are placed here. Moreover, there are some subdivisions of the dataset<br> 
```submissions/``` - collection of some of the most relevant submissions<br>
```schnetpak```- copy of the schnetpack repository, modified to work for multi-class classification<br>
```models/``` - collection of models tested<br>
- ```ChemBERTa/``` - directory with ChemBERTa model code
- ```GraphModels/``` - directory with GNN results
- ```ANN.py``` - artificial neural network workflow
- ```GraphModel.py``` - GNN model workflow
- ```SchNet.py``` - SchNet model workflow
- ```RandomForest.py``` - function which performs Random Forest multiclass Classification
- ```SVM.py``` - function which performs SVM multiclass Classification with sklearn.svm.SVC
- ```SVM_classification.py``` - function which performs SVM multiclass Classification with sklearn.linear_model.SGDClassifier
- ```XGboost.py``` - function which performs default XGBoost multiclass Classification
- ```ensemble_model.py``` - function to create ensemble model starting from "submissions" files

### Code
- ```augmentation_utils.py``` - collection of the functions used to process and transform the data, as well as perform feature engineering
- ```conversion_smiles_utils.py``` - creation of molecular representations from SMILES 
- ```data_utils.py``` - functions to import dataset and deal with imbalance
- ```similarity_scaffolds.py``` - functions to compute the number of unique scaffolds in the dataset based on Tanimoto similarity
- ```utils.py``` - Cohen's Kappa and Umap

# Environment
To install the environment, run the following command:
```
conda env create -f environment.yml -n [your_env_name]
```
and activate it via
```
conda activate [your_env_name]
```

# Executing the code
If you want to execute our code, please activate the previously installed environment first. 
You can then simply run a model by executing the corresponding .py file.
If you want to run the RF model for example, run
```
cd models/
python RandomForest.py
```

# Acknowledgement
We want to thank Andres M. Bran, Jeff Guo and Prof. Philippe Schwaller from LIAC, EPFL, for providing incredible feedback, support and guidance in this project. 
We have certainly learned a lot about applying ML methods to chemistry, the issues that arise, and possible methods on how to tackle them. 
This knowledge will help us in future projects, and we are very grateful for this exciting experience!
