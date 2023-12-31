"""
ChatGPT generated code summary

This Python script defines an AlexNet-like architecture for image classification using PyTorch. It includes functions
for training and testing the model, and a `get_best_AlexNet` function that uses WandB for hyperparameter tuning. The
script reads a YAML config file for hyperparameters, conducts a hyperparameter sweep, and saves the best model,
confusion matrix, and other results. The script demonstrates the use of the AlexNet architecture, training loops,
model saving, and visualization of results. """


import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from datamodel.datamodel import FloatImageDataset, train_test_split
from torch.utils.data import DataLoader
from train_test_suite import train_and_test_model, plot_results
from sklearn.metrics import classification_report
import yaml
from pathlib import Path
from paper_models.cm_handler import make_cm

HOME_DIRECTORY = Path.home()
SEED = 42  # for consistency - I want all of the models to get the same data


# Define the architecture of the AlexNet Clone for image classification
class AlexNet(nn.Module):
    """
    https://blog.paperspace.com/alexnet-pytorch/

    the majority of the implementation comes from here and is slightly modified for the different size
    and readability.
    """

    def __init__(self, num_classes=2,  # we're doing binary classification, so no sense using all 10
                 activation_function='relu',
                 ):
        super(AlexNet, self).__init__()

        if activation_function.lower() == "relu":
            fn = nn.ReLU()
        elif activation_function.lower() == "leaky_relu":
            fn = nn.LeakyReLU(0.1)
        else:
            fn = nn.Tanh()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),  # batch norm is a modification
            fn,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer_2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            fn,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer_3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            fn)
        self.layer_4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            fn)
        self.layer_5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            fn,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.fully_connected_1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6400, 4096),
            fn)
        self.fully_connected_2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            fn)
        self.fully_connected_3 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        # flatten here
        x = x.reshape(x.size(0), -1)
        x = self.fully_connected_1(x)
        x = self.fully_connected_2(x)
        x = self.fully_connected_3(x)

        # now you can spit out the remaining x
        return x


def get_best_AlexNet(seed=42):
    """
    this function is kind of required to make wandb work - there's likely a way to make this cleaner and not
    repeat myself, however it worked out to be faster to do it this way.

    all of the models are just slightly different enough that it was much easier time wise to go through
    and comment them like this than it was to really do effective "DRY" architecture.

    """
    with open("config/sweep.yml", "r") as yaml_file:
        sweep_config = yaml.safe_load(yaml_file)

    sweep_id = wandb.sweep(sweep=sweep_config)

    def find_best_model():

        # Initialize wandb
        wandb.init(project='AlexNet')
        config = wandb.config

        # creating the model stuff
        input_size = config.input_size  # AlexNet only accepts 224 x 224 sized images
        num_classes = 2  # this doesn't ever change either - we're doing binary classification
        learning_rate = config.learning_rate
        epochs = wandb.config.epochs

        # Create the MLP-based image classifier model
        model = AlexNet(num_classes,
                        activation_function=config.activation_function)

        path = f"{HOME_DIRECTORY}/data/0.35_reduced_then_balanced/data_{config.input_size}"
        dataset = FloatImageDataset(directory_path=path,
                                    true_folder_name="entangled", false_folder_name="not_entangled"
                                    )

        training_dataset, testing_dataset = train_test_split(dataset, train_size=0.75, random_state=SEED)
        batch_size = config.batch_size

        # create the dataloaders
        train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(testing_dataset, batch_size=batch_size)

        # Define loss function
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
        folder = f"results/transfer_learning/alexnet/{model_name}"
        if not os.path.isdir(folder):
            os.mkdir(folder)

        y_true, y_pred = history['y_true'], history['y_pred']
        classes = [str(klass).split("/")[-1] for klass in Path(path).iterdir()
                   if klass.is_dir()]

        # create a mapping from the classes to each number class
        mapping = {n: i for i, n in enumerate(classes)}
        if not os.path.exists(f'results/transfer_learning/alexnet/{wandb.run.id}'):
            os.mkdir(f'results/transfer_learning/alexnet/{wandb.run.id}')

        cr = classification_report(y_true=y_true, y_pred=y_pred)

        make_cm(
            y_actual=y_true, y_pred=y_pred, labels=[key for key in mapping.keys()],
            name=f"Modified AlexNet for Image Size {config.input_size}x{config.input_size}",
            path=folder
        )
        print(cr)

        plot_results(history, folder, title=f"Modified AlexNet for Image Size {config.input_size}x{config.input_size}")

        report = [
            model_name, "\n", cr, "\n", str(model), "\n", str(config)
        ]
        with open(f"{folder}/report.md", "w") as report_file:
            report_file.writelines(report)

        torch.save(model, f"{folder}/{model_name}.pth")

        # Log hyperparameters to wandb
        wandb.log(dict(config))

    wandb.agent(sweep_id, function=find_best_model)
