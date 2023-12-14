import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from datamodel.datamodel import FloatImageDataset, train_test_split
from torch.utils.data import DataLoader
from train_test_suite import train_and_test_model, plot_results
from sklearn.metrics import classification_report, f1_score, accuracy_score
from pathlib import Path
from paper_models.cm_handler import make_cm
import os
from statistics import stdev, mean
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from paper_models.patnet import PatNet
import pickle
from sklearn.cluster import KMeans

HOME_DIRECTORY = Path.home()
SEED = 42

# run this code to train PatNet on the full dataset!
path = f"{HOME_DIRECTORY}/data/all_data/data_224"

dataset = FloatImageDataset(directory_path=path,
                            true_folder_name="entangled", false_folder_name="not_entangled"
                            )

training_dataset, testing_dataset = train_test_split(dataset, train_size=0.8)

# hyperparameters
hyper_parameters_patnet = {'activation_function': 'leaky_relu',
                           'dropout': 0.2,
                           'hidden_sizes': 1024,
                           'learning_rate': 1e-06,
                           'optimizer': 'sgd',
                           "batch_size":32, "epochs":60}

test_dataloader = DataLoader(testing_dataset, batch_size=hyper_parameters_patnet['batch_size'])

# set up the loss function

# Set up k-fold cross-validation on the training set
k_folds = 10  # You can choose the number of folds, 5 seemed fine
kf = KFold(n_splits=k_folds, shuffle=True)

# train the encoders once - this takes forever, so we should save them when we're donePap!2903Pap!2903

# first train up a kmeans classifier on the data
training_data = [x.reshape(-1) for (x, y) in training_dataset]
print('have', len(training_data), 'images of size', set(t.shape for t in training_data))

# train the encoders once
if "encoder_0.pkl" not in os.listdir("patnet_encoders"):
    print('training k-means classifier')

    encoders = [KMeans(n_clusters=i) for i in range(2, 128, 4)]
    for i, encoder in enumerate(encoders):
        encoder.fit(training_data)

        with open(f"patnet_encoders/encoder_{i}.pkl", "wb") as encoder_file:
            pickle.dump(encoder, encoder_file)
else:
    print('loading old classifiers')
    encoders = []
    for encoder_name in os.listdir("patnet_encoders"):
        with open(f"patnet_encoders/{encoder_name}", "rb") as encoder_file:
            encoders.append(pickle.load(encoder_file))


input_size = 3 * 224 ** 2 + len(encoders)  # +1 for the extra neuron with kmeans data

accuracy = []
f1_scores = []
folds = []

# train the encoders once

# Training loop within each fold
for fold, (train_indices, val_indices) in enumerate(kf.split(training_dataset)):
    print(f"Fold {fold + 1}/{k_folds}")
    folds.append(fold + 1)
    # Split data into training and validation sets for this fold
    fold_train_dataset = torch.utils.data.Subset(dataset, train_indices)
    fold_val_dataset = torch.utils.data.Subset(dataset, val_indices)

    fold_train_dataloader = DataLoader(fold_train_dataset, batch_size=hyper_parameters_patnet['batch_size'], shuffle=True)
    fold_val_dataloader = DataLoader(fold_val_dataset, batch_size=hyper_parameters_patnet['batch_size'], shuffle=False)

    # Compute class weights
    labels = [y for x, y in fold_train_dataset]  # Implement a method to get all labels from your dataset
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    print(f'Class Weights for fold {fold}')
    print(class_weights)

    # Convert class weights to a PyTorch tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    model = PatNet(input_size,
                   [hyper_parameters_patnet['hidden_sizes'], hyper_parameters_patnet['hidden_sizes']],
                   2,
                   kmeans=encoders,
                   dropout=hyper_parameters_patnet['dropout'],
                   activation_function=hyper_parameters_patnet['activation_function'])

    optimizer = optim.Adam(model.parameters(), lr=hyper_parameters_patnet['learning_rate'])

    history = train_and_test_model(train_dataloader=fold_train_dataloader, test_dataloader=fold_val_dataloader,
                                   model=model, loss_fn=loss_fn, optimizer=optimizer,
                                   epochs=2 * hyper_parameters_patnet['epochs'],
                                   device="cpu", verbose=False, early_stopping_lookback=20)

    # save the model
    folder = f"patnet_k_folds/fold_{fold}"
    if not os.path.isdir(folder):
        os.mkdir(folder)

    y_true, y_pred = history['best_model_y_trues'], history['best_model_y_preds']
    accuracy.append(history['best_acc'])
    f1_scores.append(history["F1 Best Model"])

    best_model = history['best_model']

    cr = classification_report(y_true=y_true, y_pred=y_pred)
    make_cm(
        y_actual=y_true, y_pred=y_pred,
        name=f"PatNet for Fold {fold}",
        path=folder
    )
    print(cr)

    plot_results(history, folder, title=f"PatNet for fold {fold}")

    report = [
        fr"PatNet fold {fold}", "\n", cr, "\n", str(model), "\n"
    ]
    with open(f"{folder}/report.md", "w") as report_file:
        report_file.writelines(report)

    torch.save(best_model, f"{folder}/patnet_{fold}.pth")

mu_f1 = mean(f1_scores)
std_f1 = stdev(f1_scores)

results = {"folds": folds,
           "f1_scores": f1_scores,
           "accuracies": accuracy}

print(f1_scores)
print(accuracy)
print(f"mu f1 = {mu_f1}")
print(f"std deviation of f1 = {std_f1}")

with open('scores/patnet_scores.pkl', 'wb') as pickle_file:
    pickle.dump(results, pickle_file)
