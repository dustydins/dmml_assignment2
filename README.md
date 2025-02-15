# Data Mining & Machine Learning - Coursework 2

## Setup

#### Conda Environment:

* conda env create -f conda_env.yml
* conda activate dmmlEnv

#### Download data sets:

* Download data sets using data/get_data.sh bash script
* May need to create src/models/ directory to save models

## CLI Options

Running python main.py, the following flags/arguments should be used:

| Flag | Arguments | Description |
|:-----:|:-----:|:----:| 
| -tt or --test-type | 0-3 | Type of test set to be used |
| -c or --classifier | Classifier code (e.g. NN1) | Type of classifier to be used |
| -nv or --no-verbose | no args | Turns verbose mode off |
| -ns or --no-save | no args | If flag present, will not save model |
| -np or --no-plot | no args | If flag present, will not produce plots |


E.g. `python main.py -tt 0 -c NN1`, will run 10 fold cross-validation on the NN1 Neural network

## Test codes

* __0__: 10 fold cross-validation
* __1__: Original train/test split
* __2__: 4000 instances from training set, moved to test set
* __3__: 9000 instances from training set, moved to test set

## Classifier codes

* __NN1__: Baseline Neural Network
* __DT1__: Baseline Decision Tree
