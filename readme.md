# All Methods Broom

## Overview

This project contains Python scripts for implementing and evaluating various machine learning
models for image classification. The models include Logistic Regression, Multilayer 
Perceptron (MLP), PatNet, ResNet-18, and Support Vector Machine (SVM). The scripts 
leverage popular machine learning libraries such as scikit-learn, PyTorch, and pandas for
data manipulation and processing.

## Contents

- `logreg_model.py`: Implements a Logistic Regression model using sci-kit learn.
- `mlp_model.py`: Defines an MLP-based image classifier and performs hyperparameter tuning with wandb.
- `patnet_model.py`: Implements a PatNet model, combining KMeans clustering and MLP, with hyperparameter tuning using wandb.
- `resnet_model.py`: Defines a modified ResNet-18 classifier for transfer learning.
- `svm_model.py`: Implements an SVM model using sci-kit learn.
- `results_processing.py`: Reads and processes the results from a CSV file, generating a LaTeX table and a cleaned-up CSV file.
- `results.csv`: Sample CSV file containing model evaluation metrics.

## Usage

1. Ensure the required dependencies are installed. You can use the following:

    ```bash
    pip install -r requirements.txt
    ```

2. Run each script according to your requirements. For example:

    ```bash
    python logreg_model.py
    ```

    Ensure that the necessary datasets and configurations are in place.  Note that `main.py` will train and test all of them.

4. Examine the generated results and models in the `results` and `tex` directories.

## Results

The `results` directory and `media` directory contain model evaluations, including classification reports and confusion matrices, while the `tex` directory holds LaTeX-formatted tables.

## Requirements

- Python 3.x
- Dependencies listed in `requirements.txt`

## License

Do literally anything you want with this.