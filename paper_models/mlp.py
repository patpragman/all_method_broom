import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from datamodel.datamodel import FloatImageDataset, train_test_split
from paper_models.cm_handler import make_cm
from torch.utils.data import DataLoader
from train_test_suite import train_and_test_model, plot_results
from sklearn.metrics import classification_report
import yaml
from pathlib import Path
import os

HOME_DIRECTORY = Path.home()
SEED = 42


# Define the architecture of the MLP for image classification
class MLPImageClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 num_classes, activation_function="relu"):
        super(MLPImageClassifier, self).__init__()

        self.model_project = "Elodea MLP"

        layers = []

        layer_sizes = [input_size] + hidden_sizes + [num_classes]

        if activation_function.lower() == "relu":
            fn = nn.ReLU()
        elif activation_function.lower() == "leaky_relu":
            fn = nn.LeakyReLU(0.1)
        else:
            fn = nn.Tanh()

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(fn)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input (assuming images as input)
        x = self.mlp(x)
        return x



def get_best_mlp(seed=42):
    """
    this function is another wrapper function for wandb to do the sweep, since the models were all just a little bit
    different as I worked on them, it ended up being simpler to put these wrappers in place to initialize wandb so I
    could do them all at the same time.  Not idea, and certainly not DRY - but was faster than rewriting a bunch of my
    codebase
    """
    with open("config/mlp_sweep.yml", "r") as yaml_file:
        sweep_config = yaml.safe_load(yaml_file)

    sweep_id = wandb.sweep(sweep=sweep_config)

    def find_best_model():
        # Initialize wandb
        wandb.init(project='Elodea MLP')
        config = wandb.config

        # creating the model stuff
        input_size = 3*config.input_size**2
        hidden_sizes = [config.hidden_sizes for i in range(0, config.hidden_depth)]
        num_classes = 2  # this doesn't ever change
        learning_rate = config.learning_rate
        epochs = wandb.config.epochs

        # Create the MLP-based image classifier model
        model = MLPImageClassifier(input_size,
                                   hidden_sizes,
                                   num_classes, activation_function=config.activation_function)

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
                                       device="cpu", wandb=wandb, verbose=False, early_stopping_lookback=10)

        # save the model
        model_name = wandb.run.id
        folder = f"results/neural_networks/mlp/{model_name}"
        if not os.path.isdir(folder):
            os.mkdir(folder)

        y_true, y_pred = history['y_true'], history['y_pred']

        classes = [str(klass).split("/")[-1] for klass in Path(path).iterdir()
                   if klass.is_dir()]

        # create a mapping from the classes to each number class
        mapping = {n: i for i, n in enumerate(classes)}

        # make pretty reports below
        cr = classification_report(y_true=y_true, y_pred=y_pred)
        report = [
            model_name, "\n", cr, "\n", str(model), "\n", str(config)
        ]
        with open(f"{folder}/{model_name}_report.md", "w") as report_file:
            report_file.writelines(report)

        make_cm(
            y_actual=y_true, y_pred=y_pred, labels=[key for key in mapping.keys()],
            name=f"Multilayer Perceptron for Image Size {config.input_size}x{config.input_size}",
            path=folder
        )

        plot_results(history, folder, title=f"MLP for Image Size {config.input_size}x{config.input_size}")

        torch.save(model, f"{folder}/{model_name}.pth")  # save the model

        # Log hyperparameters to wandb
        wandb.log(dict(config))

    wandb.agent(sweep_id, function=find_best_model)
