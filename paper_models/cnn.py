import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from datamodel.datamodel import FloatImageDataset, train_test_split
from torch.utils.data import DataLoader
from train_test_suite import train_and_test_model, plot_results
from sklearn.metrics import classification_report
import yaml
from pprint import pprint
from pathlib import Path
from math import sqrt
from paper_models.cm_handler import make_cm
import os

HOME_DIRECTORY = Path.home()


def calculate_conv_layer_output_size(n, p, f, s):
    return int((n + 2*p - f)/s + 1)

class ArtisanalCNN(nn.Module):
    """
    Artisanal CNN with a few tunable hyperparameters

    """
    def __init__(self,
                 image_size,
                 layer_1_filters, layer_2_filters, layer_3_filters,  # hyperparameters
                 num_classes, activation_function="relu"):
        super(ArtisanalCNN, self).__init__()

        if activation_function.lower() == "relu":
            fn = nn.ReLU()
        elif activation_function.lower() == "leaky_relu":
            fn = nn.LeakyReLU(0.1)
        else:
            fn = nn.Tanh()

        # calculate the size of the layers
        self.layer_1_output_size = calculate_conv_layer_output_size(sqrt(image_size/3), 1, 4, 2)//2  # divided by 2 for the pooling
        self.layer_2_output_size = calculate_conv_layer_output_size(self.layer_1_output_size, 1, 4, 2)//2
        self.layer_3_output_size = calculate_conv_layer_output_size(self.layer_2_output_size, 1, 4, 2)  # no division because no pool

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=layer_1_filters,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(layer_1_filters),
            fn,
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=layer_1_filters,
                      out_channels=layer_2_filters,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(layer_2_filters),
            fn,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=layer_2_filters, out_channels=layer_3_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(layer_3_filters),
            fn)
        self.fully_connected_1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(layer_3_filters*self.layer_3_output_size**2, 1024),
            fn)
        self.fully_connected_2 = nn.Sequential(
            nn.Linear(1024, num_classes))



    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        # flatten out here
        x = x.reshape(x.size(0), -1)
        x = self.fully_connected_1(x)
        x = self.fully_connected_2(x)

        return x



def get_best_artisanal_cnn(seed=42):

    with open("config/cnn_sweep.yml", "r") as yaml_file:
        sweep_config = yaml.safe_load(yaml_file)

    sweep_id = wandb.sweep(sweep=sweep_config)

    def sweep():


        # config for wandb

        # Initialize wandb
        wandb.init(project="Artisanal CNN")
        config = wandb.config


        # creating the model stuff
        input_size = 3*config.input_size**2
        num_classes = 2  # this doesn't ever change
        learning_rate = config.learning_rate
        epochs = wandb.config.epochs

        filter_size = config.filter_sizes

        print('HYPER PARAMETERS:')
        # Create the CNN-based image classifier model
        model = ArtisanalCNN(input_size,
                             filter_size, filter_size, filter_size,
                             num_classes, activation_function=config.activation_function)

        print('Model Architecture:')
        print(model)

        path = f"{HOME_DIRECTORY}/data/0.35_reduced_then_balanced/data_{config.input_size}"

        dataset = FloatImageDataset(directory_path=path,
                                    true_folder_name="entangled", false_folder_name="not_entangled"
                                    )

        training_dataset, testing_dataset = train_test_split(dataset, train_size=0.75, random_state=seed)
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

        y_true, y_pred = history['y_true'], history['y_pred']
        classes = [str(klass).split("/")[-1] for klass in Path(path).iterdir()
                   if klass.is_dir()]

        # create a mapping from the classes to each number class and demapping
        mapping = {n: i for i, n in enumerate(classes)}
        folder = f'results/neural_networks/cnn/artisanal/{wandb.run.id}'
        if not os.path.exists(f'results/neural_networks/cnn/artisanal/{wandb.run.id}'):
            os.mkdir(f'results/neural_networks/cnn/artisanal/{wandb.run.id}')
        make_cm(
            y_actual=y_true, y_pred=y_pred, labels=[key for key in mapping.keys()],
            name=f"Artisanal CNN for Image Size {config.input_size}x{config.input_size}",
            path=f"results/neural_networks/cnn/artisanal/{wandb.run.id}"
        )
        print(classification_report(y_true=y_true, y_pred=y_pred))

        plot_results(history, folder, title=f"Artisanal CNN for Image Size {config.input_size}x{config.input_size}")


        cr = classification_report(
                y_true, y_pred, target_names=[key for key in mapping.keys()]
            )
        report = [
            f"{wandb.run.id}\n", cr, "\n", str(model), "\n", str(config)
        ]
        with open(f"{folder}/report.md", "w") as report_file:
            report_file.writelines(report)

        torch.save(model, f"results/neural_networks/cnn/artisanal/{wandb.run.id}/cnn.pth")

        # Log test accuracy to wandb

        # Log hyperparameters to wandb
        wandb.log(dict(config))

    wandb.agent(sweep_id, function=sweep)


if __name__ == "__main__":
    get_best_artisanal_cnn()


