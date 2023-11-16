# train the resnet a bit more, see what you get!
import torch
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

name = "hpuwpfhc"  # need to know precisely the model you want to train some more

model = torch.load(f"results/transfer_learning/resnet/{name}/{name}.pth")

path = f"{HOME_DIRECTORY}/data/0.35_reduced_then_balanced/data_224"

dataset = FloatImageDataset(directory_path=path,
                            true_folder_name="entangled", false_folder_name="not_entangled"
                            )

training_dataset, testing_dataset = train_test_split(dataset, train_size=0.75, random_state=SEED)
batch_size = 32

# create the dataloaders
train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
test_dataloader = DataLoader(testing_dataset, batch_size=batch_size)

# Define loss function we've used
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)


history = train_and_test_model(train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                               model=model, loss_fn=loss_fn, optimizer=optimizer, epochs=25,
                               device="cpu", wandb=None, verbose=False)

y_true, y_pred = history['y_true'], history['y_pred']
classes = [str(klass).split("/")[-1] for klass in Path(path).iterdir()
           if klass.is_dir()]

# create a mapping from the classes to each number class and demapping
mapping = {n: i for i, n in enumerate(classes)}
cr = classification_report(y_true=y_true, y_pred=y_pred)

make_cm(
    y_actual=y_true, y_pred=y_pred, labels=[key for key in mapping.keys()],
    name=f"Resnet18 for Image Size 224x224 additional training",
    path=f"models/transfer_learning/resnet/{name}"
)
print(cr)

plot_results(history, f"models/transfer_learning/resnet/{name}", title=f"ResNet18 additional training")


torch.save(model, f"models/transfer_learning/resnet/{name}/{name}_additional.pth")