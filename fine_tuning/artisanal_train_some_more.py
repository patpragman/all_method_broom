# train the resnet a bit more, see what you get!
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

from datamodel.datamodel import FloatImageDataset, train_test_split
from torch.utils.data import DataLoader
from train_test_suite import train_and_test_model, plot_results
from sklearn.metrics import classification_report
from pathlib import Path
from paper_models.cm_handler import make_cm
import os
HOME_DIRECTORY = Path.home()
SEED = 42

directory = "results/neural_networks/cnn/artisanal"

path = f"{HOME_DIRECTORY}/data/0.35_reduced_then_balanced/data_224"

dataset = FloatImageDataset(directory_path=path,
                            true_folder_name="entangled", false_folder_name="not_entangled"
                            )

training_dataset, testing_dataset = train_test_split(dataset, train_size=0.75, random_state=SEED)
batch_size = 32

# create the dataloaders
train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
test_dataloader = DataLoader(testing_dataset, batch_size=batch_size)

classes = [str(klass).split("/")[-1] for klass in Path(path).iterdir()
           if klass.is_dir()]

# create a mapping from the classes to each number class and demapping
mapping = {n: i for i, n in enumerate(classes)}

for root, _, files in os.walk(directory):

    for file in files:

        if file.endswith(".pth"):
            model_path = os.path.join(root, file)
            model = torch.load(model_path)
            name = file.replace(".pth", "_additional_training")
            print(name)

            with open(os.path.join(root, "report.md"), "r") as report_file:
                last_line = report_file.readlines()[-1]
                hyper_params = eval(last_line)
                print(hyper_params)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=hyper_params['learning_rate'], momentum=0.9)

            history = train_and_test_model(train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                               model=model, loss_fn=loss_fn, optimizer=optimizer, epochs=50,
                               device="cpu", wandb=None, verbose=False)

            y_true, y_pred = history['y_true'], history['y_pred']

            cr = classification_report(y_true=y_true, y_pred=y_pred)

            make_cm(
                y_actual=y_true, y_pred=y_pred, labels=[key for key in mapping.keys()],
                name=f"Artisanal CNN {name.replace('_', ' ')} Extra training",
                path=f"{root}"
            )
            print(cr)

            plot_results(history, f"{root}", title=f"Artisanal Additional Training")


            torch.save(model, f"{root}/{name}_additional.pth")