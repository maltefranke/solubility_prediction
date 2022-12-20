# Solubility classification of molecules

EPFL Machine Learning Course, Autumn 2022 - Class Project 2

Team members: Malte Franke, Laura Mismetti, Giacomo Mossinelli

This repository contains the code for the prediction of solubility class of compounds. In particular are distinguished compounds with:
- High solubility: Nephelometry values lower than 50000
- Medium solubility: Nephelometry values between 50000 and 100000
- Low solubility: Nephelometry values higher than 100000

# Overview
```data/``` - train and test set from Kaggle are placed here. Moreover, there are some subdivisions of the dataset 
```models/``` - collection of models tested
- ```ANN.py``` neural network
- ```GraphModel.py```
- ```SchNet.py```
- ```RandomForest.py```
- ```SVM.py```
- ```CompositeSVM.py```
- ```XGboost.py```
- ```random_forest_rcv.py```
- ```SVM_Classification_rcv.py```
- ```Xgboost_cv.py```
- ```molnet_dataloader.py```
```submissions/``` - collection of some of the most relevant submissions 

### Code
- ```Model_comparison.py```
- ```augmentation_utils.py```
- ```conversion_smiles_utils.py```
- ```count_elements.py```
- ```data_utils.py```
- ```generate_split.py```
- ```picture_molecule.png```
- ```similarity_scaffolds.py```
- ```utils.py```

# Environment
