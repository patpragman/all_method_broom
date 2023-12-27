import torch
import torchvision.transforms as transforms
from sklearn.svm import SVC
from datamodel.datamodel import FloatImageDataset, train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score
from pathlib import Path
from paper_models.cm_handler import make_cm
import os
from statistics import stdev, mean
from sklearn.model_selection import KFold
import pickle


HOME_DIRECTORY = Path.home()
SEED = 42

path = f"{HOME_DIRECTORY}/data/all_data/data_224"

dataset = FloatImageDataset(directory_path=path,
                            true_folder_name="entangled", false_folder_name="not_entangled"
                            )
# grayscale transform
grayscale_transform = transforms.Grayscale(num_output_channels=1)

training_dataset, testing_dataset = train_test_split(dataset, train_size=0.8)

# hyperparameters
hyper_parameters = {'batch_size': 32,
                    'C': 1, 'kernel': 'rbf'
                    }
test_dataloader = DataLoader(testing_dataset, batch_size=hyper_parameters['batch_size'])
del hyper_parameters['batch_size']
# set up the loss function

# Set up k-fold cross-validation on the training set
k_folds = 10  # You can choose the number of folds, 5 seemed fine
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
iterable_splitter = enumerate(kf.split(training_dataset))

accuracy = []
f1_scores = []
folds = []

# Training loop within each fold
for fold, (train_indices, val_indices) in iterable_splitter:

    classifier = SVC(**hyper_parameters)

    print(f"Fold {fold + 1}/{k_folds}")
    folds.append(fold + 1)
    # Split data into training and validation sets for this fold
    fold_train_dataset = torch.utils.data.Subset(training_dataset, train_indices)
    fold_val_dataset = torch.utils.data.Subset(training_dataset, val_indices)

    # flatten training set
    X_train = [grayscale_transform(image).flatten().numpy() for image, _ in fold_train_dataset]
    y_train = [y for _, y in fold_train_dataset]

    X_test = [grayscale_transform(image).flatten().numpy() for image, _ in fold_val_dataset]
    y_test = [y for _, y in fold_val_dataset]

    # now fit the classifier
    # fit the model with data
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    f1 = f1_score(y_test, y_pred, average="macro")
    f1_scores.append(f1)

    acc = accuracy_score(y_test, y_pred)
    accuracy.append(acc)

    # save the model
    folder = f"svm_k_folds/fold_{fold}"
    if not os.path.isdir(folder):
        os.mkdir(folder)

    cr = classification_report(y_true=y_test, y_pred=y_pred)
    make_cm(
        y_actual=y_test, y_pred=y_pred,
        name=f"SVM for Fold {fold}",
        path=folder
    )
    print(cr)

    report = [
        fr"SVM fold {fold}", "\n", str(cr), "\n", f'f1:  {f1}', "\n", f'accuracy: {acc}'
    ]
    with open(f"{folder}/report.md", "w") as report_file:
        report_file.writelines(report)

    with open(f'{folder}/svm_fold_{fold}.pkl', 'wb') as pickle_file:
        pickle.dump(classifier, pickle_file)

mu_f1 = mean(f1_scores)
std_f1 = stdev(f1_scores)

results = {"folds": folds,
           "f1_scores": f1_scores,
           "accuracies": accuracy}

print(f1_scores)
print(accuracy)
print(f"mu f1 = {mu_f1}")
print(f"std deviation of f1 = {std_f1}")

with open('scores/svm_scores.pkl', 'wb') as pickle_file:
    pickle.dump(results, pickle_file)
