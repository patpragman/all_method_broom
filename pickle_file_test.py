"""
ChatGPT generated code summary

This Python script defines functions for testing machine learning models stored as pickle files. The primary
function, `test_pickle_file`, loads a classifier from a pickle file, tests it on both original validation and unseen
test datasets, and generates classification reports and confusion matrices. Visualization of the confusion matrices
is saved in the "media" folder. The script also includes a utility function, `get_data_from_path`, to load and
preprocess image data from a given directory path. The resulting visualizations and metrics aid in assessing the
model's performance on the different datasets. """


import pickle
import os
from pathlib import Path
from skimage import io
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib
from sklearn.model_selection import train_test_split
from pytorch_file_test import save_misclassified_images_batch

matplotlib.use('Agg')  # Use the Agg backend

import matplotlib.pyplot as plt
import seaborn as sns


def get_data_from_path(path):
    classes = [str(klass).split("/")[-1] for klass in Path(path).iterdir()
               if klass.is_dir()]

    # create a mapping from the classes to each number class and demapping
    mapping = {n: i for i, n in enumerate(classes)}
    demapping = {i: n for i, n in enumerate(classes)}

    # now create an encoder
    encoder = lambda s: mapping[s]
    decoder = lambda i: demapping[i]

    # variables to hold our data
    data = []
    Y = []

    # now walk through and load the data in the containers we constructed above
    for root, dirs, files in os.walk(path):

        for file in files:
            if ".JPEG" in file.upper() or ".JPG" in file.upper() or ".PNG" in file.upper():
                key = root.split("/")[-1]

                if "cm" in file:
                    continue
                else:
                    img = io.imread(f"{root}/{file}", as_gray=True)
                    arr = np.asarray(img).reshape(224 * 224, )  # reshape into an array
                    data.append(arr)

                    Y.append(encoder(key))  # simple one hot encoding

    y = np.array(Y)
    X = np.array(data)

    # return the X, the y, and the labels for the confusion matrix
    return X, y, [key for key in mapping.keys()]


def test_pickle_file(file_path,
                     original_training_data_path,
                     unseen_dataset_path,
                     seed=42) -> dict:
    # dealing with a somewhat simpler model that's a pickle file
    with open(file_path, "rb") as file:
        classifier = pickle.load(file)

    # first, let's test the original data and create a confusion matrix
    X_original, y_original, labels_original = get_data_from_path(original_training_data_path)

    # now we've loaded all the X values into a single array
    # and all the Y values into another one, let's do a train test split
    _, X_test, __, y_test = train_test_split(X_original, y_original, test_size=0.25,
                                             random_state=seed)  # for consistency

    y_pred = classifier.predict(X_test)
    original_data_cr = classification_report(
        y_test, y_pred, target_names=labels_original, output_dict=True
    )

    original_cm = confusion_matrix(y_test, y_pred)
    # Create a heatmap to visualize the confusion matrix
    plt.clf()
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(original_cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {os.path.split(file_path)[1].split(".")[0]} on original validation data')
    plt.savefig(f'media/{os.path.split(file_path)[1].split(".")[0]}.png')

    # now test on unseen data
    X_unseen, y_unseen, labels_unseen = get_data_from_path(unseen_dataset_path)

    y_pred = classifier.predict(X_unseen)
    unseen_cm = confusion_matrix(y_unseen, y_pred)

    # Create a heatmap to visualize the confusion matrix
    plt.clf()
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(unseen_cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {os.path.split(file_path)[1].split(".")[0]} on unseen test data')
    plt.savefig(f'media/{os.path.split(file_path)[1].split(".")[0]}_unseen.png')

    unseen_data_cr = classification_report(
        y_unseen, y_pred, target_names=labels_original, output_dict=True
    )

    y_preds = []
    y_trues = []
    Xs = []
    for X, y in zip(X_unseen, y_unseen):

        y_preds.append(
            classifier.predict(X.reshape(1, -1))
        )
        y_trues.append(y)
        Xs.append(X)


    assert save_misclassified_images_batch(Xs, y_trues, y_preds,
                                               output_file=f'media/{os.path.split(file_path)[1].split(".")[0]}_missclass_unseen.png',
                                               name=os.path.split(file_path)[1].split(".")[0])

    plt.close('all')
    return original_data_cr, unseen_data_cr
