import torch
import torch.nn as nn
import torch.optim as optim
from datamodel.datamodel import FloatImageDataset, train_test_split
from torch.utils.data import DataLoader
from train_test_suite import train_and_test_model, plot_results, test
from sklearn.metrics import classification_report
from pathlib import Path
from paper_models.cm_handler import make_cm
import os
from statistics import stdev, mean
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from paper_models.resnet18 import CustomResNetClassifier
import pickle

HOME_DIRECTORY = Path.home()
SEED = 42

# run this code to train resnet-18 on the full dataset!
path = f"{HOME_DIRECTORY}/data/all_data/data_224"

dataset = FloatImageDataset(directory_path=path,
                            true_folder_name="entangled", false_folder_name="not_entangled"
                            )

training_dataset, testing_dataset = train_test_split(dataset, train_size=0.8)



# hyperparameters
hyper_parameters = {'batch_size': 32,
                    'epochs': 60,
                    'input_size': 224,
                    'learning_rate': 1e-06,
                    'optimizer': 'adam',  # not programmatic, but whatever
                    'tail_train_percentage': 0.25}
test_dataloader = DataLoader(testing_dataset, batch_size=hyper_parameters['batch_size'])

# set up the loss function

# Set up k-fold cross-validation on the training set
k_folds = 10  # You can choose the number of folds, 5 seemed fine
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
iterable_splitter = [tup for tup in enumerate(kf.split(training_dataset))]


accuracy = []
f1_scores = []
folds = []

# Training loop within each fold
for fold, (train_indices, val_indices) in iterable_splitter:

    print(f"Fold {fold + 1}/{k_folds}")
    folds.append(fold + 1)
    # Split data into training and validation sets for this fold
    fold_train_dataset = torch.utils.data.Subset(training_dataset, train_indices)
    fold_val_dataset = torch.utils.data.Subset(training_dataset, val_indices)

    fold_train_dataloader = DataLoader(fold_train_dataset, batch_size=hyper_parameters['batch_size'], shuffle=True)
    fold_val_dataloader = DataLoader(fold_val_dataset, batch_size=hyper_parameters['batch_size'], shuffle=False)

    # Compute class weights
    labels = [y for x, y in fold_train_dataset]  # Implement a method to get all labels from your dataset
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    print(f'Class Weights for fold {fold}')
    print(class_weights)

    # Convert class weights to a PyTorch tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    model = CustomResNetClassifier(tail_train_percentage=hyper_parameters['tail_train_percentage'])
    optimizer = optim.Adam(model.parameters(), lr=hyper_parameters['learning_rate'])

    history = train_and_test_model(train_dataloader=fold_train_dataloader, test_dataloader=fold_val_dataloader,
                                   model=model, loss_fn=loss_fn, optimizer=optimizer, epochs=2*hyper_parameters['epochs'],
                                   device="cpu", verbose=False, early_stopping_lookback=20)

    # save the model
    folder = f"resnet_advanced/fold_{fold}"
    if not os.path.isdir(folder):
        os.mkdir(folder)

    y_true, y_pred = history['best_model_y_trues'], history['best_model_y_preds']
    accuracy.append(history['best_acc'])
    f1_scores.append(history["F1 Best Model"])

    best_model = history['best_model']

    cr = classification_report(y_true=y_true, y_pred=y_pred)
    make_cm(
        y_actual=y_true, y_pred=y_pred,
        name=f"Resnet18 for Fold {fold}",
        path=folder
    )
    print(cr)

    plot_results(history, folder, title=f"ResNet18 for fold {fold}")

    report = [
        fr"resnet 18 fold {fold}", "\n", cr, "\n", str(model), "\n", f"Best Epoch:  {history['best_epoch']}", "\n"
    ]
    with open(f"{folder}/report.md", "w") as report_file:
        report_file.writelines(report)

    torch.save(best_model, f"{folder}/resnet18_fold_{fold}.pth")

# ok, let's save some information about these
mu_f1 = mean(f1_scores)
std_f1 = stdev(f1_scores)


results = {"folds": folds,
            "f1_scores": f1_scores,
           "accuracies": accuracy}

print(f1_scores)
print(accuracy)
print(f"mu f1 = {mu_f1}")
print(f"std deviation of f1 = {std_f1}")

with open('scores/resnet_scores.pkl', 'wb') as pickle_file:
    pickle.dump(results, pickle_file)


# now that we've saved the "best" model from each group let's get the model with the best F1 score, then train that
# on the remaining data

#  best_index = torch.argmax(f1_scores) # - this may be slightly faster?  Not sure.
best_index = max(enumerate(f1_scores), key=lambda x: x[1])[0]
best_folder = f"resnet_advanced/fold_{best_index}"
best_model_path = f"{best_folder}/resnet18_fold_{best_index}.pth"

# Load the best most performant model
best_model = torch.load(best_model_path)

# let's make a folder to stash the best results in
folder = f"resnet_advanced/best_model_advanced_training"
if not os.path.isdir(folder):
    os.mkdir(folder)



# now let's go back, and give this model *all* the training data that remains to see how it performs!
for fold, (train_indices, val_indices) in iterable_splitter:
    print('training best model on remaining training data!')
    best_model.train()
    if fold == best_index:
        # don't retrain on data we've already trained on!
        continue

    # Split data into training and validation sets for this fold
    fold_train_dataset = torch.utils.data.Subset(training_dataset, train_indices)
    fold_val_dataset = torch.utils.data.Subset(training_dataset, val_indices)

    fold_train_dataloader = DataLoader(fold_train_dataset, batch_size=hyper_parameters['batch_size'], shuffle=True)
    fold_val_dataloader = DataLoader(fold_val_dataset, batch_size=hyper_parameters['batch_size'], shuffle=False)

    # Compute class weights
    labels = [y for x, y in fold_train_dataset]  # Implement a method to get all labels from your dataset
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    print(f'Class Weights for fold {fold}')
    print(class_weights)

    # Convert class weights to a PyTorch tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    optimizer = optim.Adam(best_model.parameters(), lr=hyper_parameters['learning_rate'])

    history = train_and_test_model(train_dataloader=fold_train_dataloader, test_dataloader=fold_val_dataloader,
                                   model=best_model, loss_fn=loss_fn, optimizer=optimizer, epochs=2*hyper_parameters['epochs'],
                                   device="cpu", verbose=False, early_stopping_lookback=20)


    y_true, y_pred = history['best_model_y_trues'], history['best_model_y_preds']
    accuracy.append(history['best_acc'])
    f1_scores.append(history["F1 Best Model"])

    # only pull the most performant model out - for instance if you overtrained, don't take the overfit models
    best_model = history['best_model']

    cr = classification_report(y_true=y_true, y_pred=y_pred)
    print(cr)

    plot_results(history, folder, title=f"Best ResNet18 for fold {fold}")

    report = [
        fr"resnet 18 fold {fold}", "\n", cr, "\n", str(model), "\n"
    ]
    with open(f"{folder}/report_{fold}.md", "w") as report_file:
        report_file.writelines(report)


torch.save(best_model, f"{folder}/resnet18_best.pth")
best_model.eval()

# now test it on entirely unseen data, start by getting the class weigths
labels = [y for x, y in testing_dataset]
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
print(f'Class Weights for testing dataset:')
print(class_weights)

# Convert class weights to a PyTorch tensor
class_weights = torch.tensor(class_weights, dtype=torch.float32)
loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
correct, test_loss, y_pred, y_true = test(test_dataloader,
                                          best_model,
                                          loss_fn, "cpu")
make_cm(
    y_actual=y_true, y_pred=y_pred,
    name=f"Best ResNet18 with All Data",
    path=folder
)

