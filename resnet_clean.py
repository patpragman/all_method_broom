import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from datamodel.datamodel import FloatImageDataset, train_test_split
from torch.utils.data import DataLoader
from train_test_suite import train_and_test_model, plot_results, test
from sklearn.metrics import classification_report, f1_score, accuracy_score
import yaml
from pathlib import Path
from paper_models.cm_handler import make_cm
import os
from paper_models.resnet18 import CustomResNetClassifier

HOME_DIRECTORY = Path.home()


if __name__ == "__main__":
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    # run this code to train resnet-18 on the full dataset!
    path = f"{HOME_DIRECTORY}/clean_all_data/data_224"

    dataset = FloatImageDataset(directory_path=path,
                                true_folder_name="entangled", false_folder_name="not_entangled"
                                )

    training_dataset, testing_dataset = train_test_split(dataset, train_size=0.8)

    # hyperparameters
    hyper_parameters = {'batch_size': 32,
                        'epochs': 2*60,
                        'input_size': 224,
                        'learning_rate': 1e-06,
                        'optimizer': 'adam',  # not programatic, but whatever
                        'tail_train_percentage': 0.25}
    test_dataloader = DataLoader(testing_dataset, batch_size=hyper_parameters['batch_size'])

    # finally, let's train a model on on the whole train clean dataset, then evaluate that:
    full_train_data_loader = DataLoader(training_dataset, batch_size=hyper_parameters['batch_size'])

    # Compute class weights
    labels = [y for x, y in training_dataset]  # Implement a method to get all labels from your dataset
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    print(f'Class Weights for fold full training data')
    print(class_weights)

    # Convert class weights to a PyTorch tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    model = CustomResNetClassifier(tail_train_percentage=hyper_parameters['tail_train_percentage'])
    optimizer = optim.Adam(model.parameters(), lr=hyper_parameters['learning_rate'])

    history = train_and_test_model(train_dataloader=full_train_data_loader, test_dataloader=test_dataloader,
                                   model=model, loss_fn=loss_fn, optimizer=optimizer,
                                   epochs=2*hyper_parameters['epochs'],
                                   device="cpu", verbose=False, early_stopping_lookback=20)

    # save the model
    folder = f"resnet_clean_data"
    if not os.path.isdir(folder):
        os.mkdir(folder)

    best_model = history['best_model']

    # run test code on wholy unseen data
    correct, test_loss, y_pred, y_true = test(
        test_dataloader,
        best_model,
        loss_fn,
        "cpu",
    )


    cr = classification_report(y_true=y_true, y_pred=y_pred)
    make_cm(
        y_actual=y_true, y_pred=y_pred,
        name=f"ResNet-18 on Clean 224x224 Images with Unseen Test Data",
        path=folder
    )
    print(cr)

    plot_results(history, folder, title=f"ResNet-18 on Clean 224x224 Images at epoch {history['best_epoch']}")

    report = [
        f"Best Epoch: {history['best_epoch']}", "\n",
        fr"resnet 18 clean", "\n", cr, "\n", str(model), "\n"
    ]
    with open(f"{folder}/report.md", "w") as report_file:
        report_file.writelines(report)

    torch.save(best_model, f"{folder}/resnet18_clean.pth")


