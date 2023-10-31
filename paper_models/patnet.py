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
from sklearn.cluster import KMeans
from torchsummary import summary

HOME_DIRECTORY = Path.home()
SEED = 42


# Define the architecture of the MLP for image classification
class PatNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 num_classes,
                 kmeans,
                 dropout,
                 activation_function="relu"):
        super(PatNet, self).__init__()

        self.kmeans = kmeans
        self.model_project = "PatNet"
        self.dropout_value = dropout

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
                layers.append(nn.Dropout(self.dropout_value))
                layers.append(fn)


        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 3*224*224).float()  # Flatten the input
        kmeans_prediction = torch.Tensor(self.kmeans.predict(x)).view(-1, 1)  # kmeans off of that image
        x = torch.cat((x, kmeans_prediction), dim=1)
        x = self.mlp(x)
        return x


def get_best_patnet(seed=42):
    with open("config/patnet_sweep.yml", "r") as yaml_file:
        sweep_config = yaml.safe_load(yaml_file)

    sweep_id = wandb.sweep(sweep=sweep_config)

    def find_best_model():
        # config for wandb

        # Initialize wandb
        wandb.init(project='Elodea PatNet')
        config = wandb.config

        # creating the model stuff
        input_size = 3 * config.input_size ** 2 + 1  # +1 for the extra neuron with kmeans data
        hidden_sizes = [config.hidden_sizes for i in range(0, config.hidden_depth)]
        num_classes = 2  # this doesn't ever change
        learning_rate = config.learning_rate
        epochs = wandb.config.epochs

        # get the data
        path = f"{HOME_DIRECTORY}/data/0.35_reduced_then_balanced/data_{config.input_size}"

        dataset = FloatImageDataset(directory_path=path,
                                    true_folder_name="entangled", false_folder_name="not_entangled"
                                    )

        training_dataset, testing_dataset = train_test_split(dataset, train_size=0.75, random_state=SEED)
        batch_size = config.batch_size

        # create the dataloaders
        train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(testing_dataset, batch_size=batch_size)


        # first train up a kmeans classifier on the data
        print('training k-means classifier')
        training_data = [x.reshape(-1) for (x, y) in training_dataset]
        print('have', len(training_data), 'images of size', set(t.shape for t in training_data))
        kmeans = KMeans(n_clusters=config.kmeans_size, random_state=42)
        kmeans.fit(training_data)


        # Create the MLP-based image classifier model
        model = PatNet(input_size,
                       hidden_sizes,
                       num_classes,
                       kmeans=kmeans,
                       dropout=config.dropout,
                       activation_function=config.activation_function)

        print(summary(model, input_size=(3, 224, 224), batch_size=batch_size))
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

        if history['F1 Best Model'] > 0.75:
            # I'm tired of saving hundreds of models only some of which are any good
            # narrow this down for me

            # save the model
            model_name = wandb.run.id
            print(f"model: {model_name} worth further investigation")
            print("F1 Best Model", history['F1 Best Model'])
            folder = f"results/modified_patnet/{model_name}"
            if not os.path.isdir(folder):
                os.mkdir(folder)

            y_true, y_pred = history['y_true'], history['y_pred']
            best_model = history['best_model']
            best_y_preds = history['best_model_y_preds']

            classes = [str(klass).split("/")[-1] for klass in Path(path).iterdir()
                       if klass.is_dir()]

            # create a mapping from the classes to each number class
            mapping = {n: i for i, n in enumerate(classes)}

            # make a report classification report of the last model to train, and the best model trained
            cr = classification_report(y_true=y_true, y_pred=y_pred)
            report = [
                f'last_{model_name}', "\n", cr, "\n", str(model), "\n", str(config)
            ]
            with open(f"{folder}/last_{model_name}_report.md", "w") as report_file:
                report_file.writelines(report)

            cr = classification_report(y_true=history['best_model_y_trues'], y_pred=best_y_preds)
            report = [
                f'best_{model_name}', "\n", cr, "\n", str(best_model), "\n", str(config)
            ]
            with open(f"{folder}/best_{model_name}_report.md", "w") as report_file:
                report_file.writelines(report)

            # make a cm for the last model trained
            make_cm(
                y_actual=y_true, y_pred=y_pred, labels=[key for key in mapping.keys()],
                name=f"Dropout PatNet at epoch {history['ending_epoch']}",
                path=folder
            )

            # now one for the best model
            make_cm(
                y_actual=history['best_model_y_trues'], y_pred=history['best_model_y_preds'], labels=[key for key in mapping.keys()],
                name=f"PatNet at epoch {history['best_epoch']}",
                path=folder
            )

            plot_results(history, folder, title=f"PatNet for Image Size {config.input_size}x{config.input_size}")

            torch.save(history['best_model'], f"{folder}/best_{model_name}.pth")
            torch.save(model, f"{folder}/last_{model_name}.pth")
        else:
            print('model not worth saving')

        # Log hyperparameters to wandb
        wandb.log(dict(config))

    wandb.agent(sweep_id, function=find_best_model)

if __name__ == "__main__":
    get_best_patnet()