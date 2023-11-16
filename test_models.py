"""
we test all of the models here.

make sure all the models are in an appropriate saved subfolders of the "results" folder, then iterate through all the
subfolders of that, snagging all the pytorch models.  put the results into a .csv file for us to work with in
"generate_results_table.py"

"""

"""
ChatGPT generated code summary

This Python script is designed to test and evaluate machine learning models stored in the "results" folder. The 
script iterates through subfolders, identifying models in either pickle (.pkl) or PyTorch (.pth) format. For each 
model, it performs evaluation using either the `test_pickle_file` function (for .pkl files) or the 
`predict_with_model` function (for .pth files). The evaluation results, specifically precision, recall, and F1-score 
for the "entangled" and "not_entangled" classes, are stored in a dictionary named `classification_reports`. Finally, 
the script organizes this information into a pandas DataFrame and exports it to a CSV file named "results.csv". The 
resulting CSV file is intended for further analysis in the "generate_results_table.py" script. """



import os
from pathlib import Path
import torch
from torch import nn
from datamodel.datamodel import FloatImageDataset
from pickle_file_test import test_pickle_file
from pytorch_file_test import predict_with_model
import pandas as pd
from paper_models.patnet import PatNet  # I shouldn't have to include this, but for some reason dependency issues required it

# load the path to the prior seen data so we can make pretty charts!
original_training_data_path = f"{Path.home()}/data/0.35_reduced_then_balanced/data_224"

# load the path to the unseen data:
unseen_data_folder_path = f"{Path.home()}/data/unseen/data_224"
model_folder_path = "results"

# batch size 32
BATCH_SIZE = 32
SEED = 42

# loss function
loss_function = nn.CrossEntropyLoss()

unseen_dataset = FloatImageDataset(unseen_data_folder_path,
                                   true_folder_name="entangled", false_folder_name="not_entangled")

original_data = FloatImageDataset(original_training_data_path,
                                  true_folder_name="entangled", false_folder_name="not_entangled")

# loop through all the models in the model folder, then if it's a model, test it!
classification_reports = {}
models = []
sizes = []

for root, _, files in os.walk(model_folder_path):
    """
    walk through the directory containing all the various models produced by hyperparameter searching
    """
    for model_file_name in files:

        if model_file_name.endswith(".pkl"):
            # .pkl files are sklearn files - they should use the "test_pickle_file" function
            print(f'Evaluating: {model_file_name}')
            crs = test_pickle_file(os.path.join(root, model_file_name),
                                   original_training_data_path,
                                   unseen_data_folder_path)
            model_name = model_file_name.replace(".pkl", "")
            data = {"original_training_data": crs[0], "unseen_training_data": crs[1],
            }
        elif model_file_name.endswith(".pth"):
            # .pth models are pytorch models - they need to use pytorch stuff
            print(f'Evaluating: {model_file_name}')
            model = torch.load(os.path.join(root, model_file_name))
            model_name = f"{os.path.split(root)[-2].split('/')[2]}_{os.path.split(model_file_name)[1].split('.')[0]}"
            try:
                crs = predict_with_model(model,
                                         original_data,
                                         unseen_dataset,
                                         loss_function,
                                         model_name=model_name,
                                         batch_size=BATCH_SIZE)
            except IndexError as index_error:
                # when you get a model that only guesses one type this error occurs - we don't want that model
                # so we will ignore the error, but we should print something to the console so the user knows what model
                # to go look for
                print('check:', model_name, 'probably had all one category')
                continue

            data = {"original_training_data": crs[0], "unseen_training_data": crs[1],}
        else:
            print(f'{model_file_name} is not a model...')
            continue

        classification_reports[model_name] = data['unseen_training_data']


# now we're going to put the output into a .csv file we can evaluate
reformatted = {}
sub_keys = ['not_entangled', 'entangled']
targets = ['precision', 'recall', 'f1-score']
for key in classification_reports:
    reformatted[key] = {}

    for subkey in sub_keys:
        for target in targets:
            reformatted[key][f'{subkey}_{target}'] = classification_reports[key][subkey][target]

df = pd.DataFrame(reformatted)
print(df)
df.to_csv("results.csv")

# this .csv file is manipulated in the "generate results label function"

