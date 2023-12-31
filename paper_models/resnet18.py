"""
ChatGPT generated code summary

This Python script defines a custom ResNet18 classifier for image classification using PyTorch. It includes
functionalities for hyperparameter tuning through a sweep with WandB. The script loads a pre-trained ResNet18 model,
freezes a specified percentage of layers, and appends custom fully connected layers for classification. Training and
evaluation are performed on a specified dataset, and the results are saved, logged, and visualized.

Noteworthy points:
1. The custom ResNet18 classifier freezes a percentage of layers during training.
2. The script uses WandB for hyperparameter tuning, logging, and visualization.
3. The classification report and confusion matrix are generated and saved.
4. The best model, training history, and reports are saved for analysis.

Ensure you have the required libraries and dataset paths configured before running this script.
"""


import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import wandb
from datamodel.datamodel import FloatImageDataset, train_test_split
from torch.utils.data import DataLoader
from train_test_suite import train_and_test_model, plot_results
from sklearn.metrics import classification_report, f1_score, accuracy_score
import yaml
from pathlib import Path
from paper_models.cm_handler import make_cm
import os
from statistics import stdev, mean

HOME_DIRECTORY = Path.home()
SEED = 42




class CustomResNetClassifier(nn.Module):
    """
    Resnet18 tweaked classifer

    """

    def __init__(self,
                 tail_train_percentage=0.25,
                 number_of_classes=2):
        super(CustomResNetClassifier, self).__init__()

        self.resnet18 = models.resnet18(pretrained=True)

        # now freeze the last tail_train_percentage of layers
        depth = len(list(self.resnet18.parameters()))
        transition_level = depth * (1 - tail_train_percentage)
        for i, param in enumerate(self.resnet18.parameters()):
            if i <= transition_level:
                param.requires_grad = False

        # remove the last layer - we don't want it to classify, that's for us to do
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])

        self.flat_layer_1 = nn.Linear(512, 512)
        self.flat_layer_2 = nn.Linear(512, 512)
        self.flat_layer_3 = nn.Linear(512, 512)

        self.classifier = nn.Linear(512, number_of_classes)

    def forward(self, x):
        # Extract features using the ResNet-18 backbone
        features = self.resnet18(x)

        # Flatten the features if needed (e.g., if you have spatial dimensions)
        features = features.view(features.size(0), -1)

        features = self.flat_layer_1(features)
        features = self.flat_layer_2(features)
        features = self.flat_layer_3(features)

        # Apply the fully connected layer to get class logits
        logits = self.classifier(features)

        return logits



def get_best_resnet(seed=42):
    """
    basically the same as all the other ones - this is a wrapper function for wandb convenience


    :param seed: unused but kept in for consistency
    :return:
    """
    with open("config/resnet_sweep.yml", "r") as yaml_file:
        yml = yaml.safe_load(yaml_file)
        project_name = yml['project']
        sweep_config = yml

    sweep_id = wandb.sweep(sweep=sweep_config)
    def find_best_model():
        # config for wandb

        # Initialize wandb
        wandb.init(project="ResNet18")
        config = wandb.config

        # creating the model stuff
        learning_rate = config.learning_rate
        epochs = wandb.config.epochs

        # create a custom resnet, retraining the last percentage of the layers
        model = CustomResNetClassifier(tail_train_percentage=config.tail_train_percentage)

        path = f"{HOME_DIRECTORY}/data/0.35_reduced_then_balanced/data_{config.input_size}"

        dataset = FloatImageDataset(directory_path=path,
                                    true_folder_name="entangled", false_folder_name="not_entangled"
                                    )

        training_dataset, testing_dataset = train_test_split(dataset, train_size=0.75, random_state=SEED)
        batch_size = config.batch_size

        # create the dataloaders
        train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(testing_dataset, batch_size=batch_size)

        # Define loss function we've used
        loss_fn = nn.CrossEntropyLoss()

        # optimzer parsing logic:
        if config.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        history = train_and_test_model(train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                                       model=model, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs,
                                       device="cpu", wandb=wandb, verbose=False)

        # save the model
        model_name = wandb.run.id
        folder = f"results/transfer_learning/resnet/{model_name}"
        if not os.path.isdir(folder):
            os.mkdir(folder)

        y_true, y_pred = history['y_true'], history['y_pred']
        classes = [str(klass).split("/")[-1] for klass in Path(path).iterdir()
                   if klass.is_dir()]

        # create a mapping from the classes to each number class and demapping
        mapping = {n: i for i, n in enumerate(classes)}
        if not os.path.exists(f'results/transfer_learning/resnet/{wandb.run.id}'):
            os.mkdir(f'results/transfer_learning/resnet/{wandb.run.id}')

        cr = classification_report(y_true=y_true, y_pred=y_pred)

        make_cm(
            y_actual=y_true, y_pred=y_pred, labels=[key for key in mapping.keys()],
            name=f"Resnet18 for Image Size {config.input_size}x{config.input_size}",
            path=f"results/transfer_learning/resnet/{wandb.run.id}"
        )
        print(cr)

        plot_results(history, folder, title=f"ResNet18 for Image Size {config.input_size}x{config.input_size}")

        report = [
            model_name, "\n", cr, "\n", str(model), "\n", str(config)
        ]
        with open(f"{folder}/report.md", "w") as report_file:
            report_file.writelines(report)

        torch.save(model, f"{folder}/{model_name}.pth")

        # Log hyperparameters to wandb
        wandb.log(dict(config))

    wandb.agent(sweep_id, function=find_best_model)


if __name__ == "__main__":
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    # run this code to train resnet-18 on the full dataset!
    path = f"{HOME_DIRECTORY}/data/all_data/data_224"

    dataset = FloatImageDataset(directory_path=path,
                                true_folder_name="entangled", false_folder_name="not_entangled"
                                )

    training_dataset, testing_dataset = train_test_split(dataset, train_size=0.8, random_state=42)



    # hyperparameters
    hyper_parameters = {'batch_size': 32,
                        'epochs': 60,
                        'input_size': 224,
                        'learning_rate': 1e-06,
                        'optimizer': 'adam',  # not programatic, but whatever
                        'tail_train_percentage': 0.25}
    test_dataloader = DataLoader(testing_dataset, batch_size=hyper_parameters['batch_size'])


    # finally, let's train a model on on the whole training set, then evaluate that:

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
    folder = f"resnet_advanced/final"
    if not os.path.isdir(folder):
        os.mkdir(folder)

    y_true, y_pred = history['best_model_y_trues'], history['best_model_y_preds']

    best_model = history['best_model']

    cr = classification_report(y_true=y_true, y_pred=y_pred)
    make_cm(
        y_actual=y_true, y_pred=y_pred,
        name=f"ResNet-18 on full training dataset",
        path=folder
    )
    print(cr)

    plot_results(history, folder, title=f"ResNet-18 on full training dataset at epoch {history['best_epoch']}")

    report = [
        f"Best Epoch: {history['best_epoch']}", "\n",
        fr"resnet 18 full", "\n", cr, "\n", str(model), "\n"
    ]
    with open(f"{folder}/report.md", "w") as report_file:
        report_file.writelines(report)

    torch.save(best_model, f"{folder}/resnet18_complete.pth")


