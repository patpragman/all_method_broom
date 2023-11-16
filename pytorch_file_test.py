import torch
from train_test_suite import test
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from skimage import io
import torch
import matplotlib
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from datamodel.datamodel import train_test_split
matplotlib.use('Agg')  # Use the Agg backend

import matplotlib.pyplot as plt
import numpy as np


def save_misclassified_images_batch(image_batch,
                                    actual_labels,
                                    predicted_labels,
                                    output_file, name):

    false_positives = []
    false_negatives = []

    for image, y_true, y_pred in zip(image_batch, actual_labels, predicted_labels):
        if (y_true == 0) and (y_pred == 1):
            # false positive
            false_positives.append(image)
        elif (y_true == 1) and (y_pred == 0):
            # false negative
            false_negatives.append(image)

    if len(false_positives) + len(false_negatives) < 8:
        return False

    # make a list of misclassified images that's 8 long
    misclassified = false_positives[0:4] + false_negatives[0:4]

    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(f"Misclassified Images: {name}", fontsize=16)

    for i, ax in enumerate(axes.ravel()):
        image = misclassified[i]
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            image = image / 255.0
            image = np.transpose(image, (1, 2, 0))
            cmap = plt.get_cmap()
        else:
            # not that kind, it's a numpy array, reshape it
            image = image / 255.0
            image = image.reshape(224, 224)
            cmap = plt.get_cmap("gray")

        if i <= 3:
            actual = 'Not Entangled'
            prediction = 'Entangled'
        else:
            actual = 'Entangled'
            prediction = 'Not Entangled'

        ax.imshow(image, cmap=cmap)
        ax.set_title(f"Actual: {actual}\nPredicted: {prediction}", fontsize=8)
        ax.axis('off')

    plt.savefig(output_file)
    plt.close()
    plt.clf()

    return True


def predict_with_model(model,
                       original_dataset,
                       unseen_dataset,
                       loss_fn,
                       model_name,
                       device="cpu",
                       batch_size=32):
    # must resplit the original dataset exactly how it was so we can get the testing data
    training_dataset, testing_dataset = train_test_split(original_dataset, train_size=0.75, random_state=42)
    original_dataloader = DataLoader(testing_dataset, batch_size=batch_size)
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=batch_size)

    original_correct, original_loss, original_y_true, original_y_pred = test(original_dataloader, model, loss_fn,
                                                                             device=device)
    unseen_correct, unseen_loss, unseen_y_true, unseen_y_pred = test(unseen_dataloader, model, loss_fn, device=device)

    classes = [str(klass).split("/")[-1] for klass in Path(unseen_dataset.directory_path).iterdir()
               if klass.is_dir()]

    # create a mapping from the classes to each number class and demapping
    mapping = {n: i for i, n in enumerate(classes)}
    demapping = {i: n for i, n in enumerate(classes)}

    original_cr = classification_report(
        original_y_true, original_y_pred, target_names=[key for key in mapping.keys()], output_dict=True
    )

    unseen_cr = classification_report(
        unseen_y_true, unseen_y_pred, target_names=[key for key in mapping.keys()], output_dict=True
    )

    original_cm = confusion_matrix(original_y_true, original_y_pred)

    # Create a heatmap to visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)

    sns.heatmap(original_cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name} on original validation data')
    plt.savefig(f'media/{model_name}.png')

    # Create a heatmap to visualize the confusion matrix
    unseen_cm = confusion_matrix(unseen_y_true, unseen_y_pred)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(unseen_cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name} on unseen test data')
    plt.savefig(f'media/{model_name}_unseen.png')

    model.eval()

    Xs = []
    y_preds = []
    y_trues = []
    for X, y in unseen_dataloader:

        prediction = model(X)

        y_true = y.cpu().numpy().tolist()  # Convert the tensor to a CPU numpy array
        y_pred = prediction.argmax(1).cpu().numpy().tolist()  # Convert the tensor to a CPU numpy array

        for image, y_true, y_pred in zip(X, y_true, y_pred):
            Xs.append(image)
            y_trues.append(y_true)
            y_preds.append(y_pred)

    assert save_misclassified_images_batch(Xs, y_trues, y_preds,
                                               output_file=f'media/{model_name}_missclass_unseen.png',
                                               name=model_name)

    plt.close('all')
    return original_cr, unseen_cr
