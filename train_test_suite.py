"""
This is a simple library with some convenience libraries to train my PyTorch models.
Basically, I built some code that duplicates default functionality in Keras, but I
needed the flexibility of PyTorch.
"""

"""
ChatGPT generated code summary

This Python script provides a simple library for training PyTorch models. It includes functions for training, 
testing, and evaluating models. The `train` function trains a model using provided data, loss function, 
and optimizer. The `test` function evaluates the model on a test set and returns metrics like accuracy and loss. The 
`train_and_test_model` function combines training and testing, also supporting optional early stopping. Additionally, 
the script contains a utility function, `plot_results`, for visualizing training and testing results. This library is 
designed to offer flexibility in PyTorch while incorporating some conveniences inspired by Keras. """



import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import copy
from statistics import mean

def train(dataloader: DataLoader,
          model: nn.Module,
          loss_fn: nn.modules.loss._Loss,  # type hints are stupid and difficult for this
          optimizer: torch.optim.Optimizer,
          device: str,
          verbose: bool = False) -> float:
    """

    :param dataloader: a pytorch dataloader containing the training data
    :param model: a pytorch Module that is the instantiated neural network
    :param loss_fn: the PyTorch loss function you're using
    :param optimizer: similarly, the PyTorch optimizer you're using (for instance ADAM)
    :param device: the device you're going to do the training on as a string "cpu" or "gpu" basically
    :param verbose: flag to turn on some text output during training - kind of obnoxious but was useful for debugging
    :return:
    """
    size = len(dataloader.dataset)
    model.train()  # note we didn't define this, it must be in the parent class

    for batch, (X, y) in enumerate(dataloader):
        # main training loop
        X, y = X.to(device), y.to(device)  # send the work to the device

        prediction = model(X)  # compute the prediction
        loss = loss_fn(prediction, y)  # compute the loss value

        # now do back prop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # zero the gradients

        if batch % 100 == 0 and verbose:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current: >5d}/{size:>5d}]")

    return loss.item()  # return the final loss


def test(dataloader: DataLoader,
         model: nn.Module,
         loss_fn: nn.modules.loss._Loss,  # type hints are stupid and difficult for this
         device: str,
         verbose: bool = False) -> tuple:
    """
    :param dataloader: the test dataloader
    :param model: the pytorch model you're testing
    :param loss_fn: the loss function you're using
    :param device: a string containing "cuda" or "cpu" depending on what you're trying to train on
    :param verbose: a flag to turn on some obnoxious console output - very useful for debugging
    :return:
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for X, y in dataloader:
            # main testing loop
            X, y = X.to(device), y.to(device)
            prediction = model(X)

            y_true = y.cpu().numpy().tolist()  # Convert the tensor to a CPU numpy array
            y_pred = prediction.argmax(1).cpu().numpy().tolist()  # Convert the tensor to a CPU numpy array

            y_pred_list.append(y_pred)
            y_true_list.append(y_true)

            test_loss += loss_fn(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()  # wtf is this doing?!

    test_loss /= num_batches
    correct /= size

    if verbose:
        print(f"Test Error: \n Accuracy: {(100 * correct): >0.1f}%, avg loss: {test_loss: >8f} \n")

    # sum(a_nested_list, []) flattens the list
    # we return a lot here - this is not what I function should do, but as the project grew I kept finding new
    # stuff I needed to pass back as part of the optimization process.  This needs a thorough rewrite at some point
    return correct, test_loss, sum(y_pred_list, []), sum(y_true_list, [])


def train_and_test_model(
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.modules.loss._Loss,  # type hints are stupid and difficult for pytorch - but this is your loss fn
        optimizer: torch.optim.Optimizer,
        device: str,
        epochs: int = 10,
        verbose: bool = False,
        wandb=None,
        early_stopping_lookback=10,
) -> dict:
    """
    this function duplicates some of the functionality found automagically in Keras - we pass a training set,
    and a validation set to this, and get back a dictionary of "history" containing a bunch of useful information
    we can use later - needs a complete refactoring at some point.

    :param train_dataloader: the pytorch dataloader object containing the training set
    :param test_dataloader: the pytorch dataloader object containing the validation data (should refactor nomenclature)
    :param model: the pytorch model you want to train
    :param loss_fn: the loss function - typically categorical cross entropy, but you do you
    :param optimizer: pytorch optimizer, like adam or sgd
    :param device: the device you want to train on, this is a string with "cuda" or "gpu"
    :param epochs: how many epochs to train - this is an integer
    :param verbose: flag for turning on obnoxious console logging - helpful for debug, but terrible otherwise
    :param wandb: a wandb object if you want to send stuff to wandb
    :param early_stopping_lookback: how many epochs you want to look back for early stopping.
    :return:
    """
    # instantiate a bunch of useful variables
    training_losses = []
    testing_losses = []
    testing_accuracies = []
    epoch = []

    best_epoch = -1
    best_model = None
    best_y_trues = []
    best_predictions = []
    best_f1 = -1
    best_acc = -1

    # tqdm performs poorly with verbose mode if you're debugging a model, so only use tqdm for
    # a progress bar if you're not using verbose logging.
    iterator = tqdm(range(epochs)) if not verbose else range(epochs)
    for t in iterator:
        if verbose:
            print(f"Epoch {t + 1}:\n ----------------------------------------------------------")

        training_loss = train(train_dataloader,
                              model,
                              loss_fn,
                              optimizer,
                              device,
                              verbose=verbose)

        training_losses.append(training_loss)

        # we get a bunch of information back from test that we use to evaluate training - this needs to be refactored
        # as this is pretty tightly coupled, still, sufficient for now (I know, there's no such thing as "good enough
        # for now". :(
        test_acc, test_loss, y_pred_list, y_true_list = test(test_dataloader, model, loss_fn, device, verbose=verbose)
        testing_losses.append(test_loss)
        testing_accuracies.append(test_acc)
        epoch.append(t)

        f1 = f1_score(y_true_list, y_pred_list, average="macro")
        if best_f1 < f1:
            best_acc = test_acc
            best_f1 = f1
            best_epoch = t
            # copy the model, and her predictions for that epoch - use deep copy
            best_model = copy.deepcopy(model)
            best_predictions = copy.deepcopy(y_pred_list)
            best_y_trues = copy.deepcopy(y_true_list)

        if not verbose:
            # this let's us adjust the tqdm progress bar output
            iterator.set_description(
                f"Training Loss: {training_loss:.2f} Testing Loss: {test_loss:.2f} Accuracy {test_acc:.2f}"
            )

        if wandb:
            # this is for logging at every epoch
            wandb.log({"training loss": training_loss})
            wandb.log({"testing loss": test_loss})
            wandb.log({"accuracy": test_acc})
            wandb.log({"epoch": t})
            wandb.log({"F1": f1_score(best_y_trues, best_predictions)})

        # early stopping off of loss
        if len(testing_losses) > early_stopping_lookback > 0:
            # if you set early_stopping_lookback to less than zero you will not perform early stopping
            if min(testing_losses) in testing_losses[-early_stopping_lookback:]:
                # if the smallest testing loss is in the lookback window, keep going
                continue
            elif 3*mean(testing_losses[-early_stopping_lookback:]) < mean(training_losses[-early_stopping_lookback:]):
                # let's look for overfitting now, if the 3x the mean of the testing losses during the lookback
                # is less than the mean of the training loss during the look back, break out of the loop
                break
            else:
                # otherwise, break out of the loop
                break

    last_epoch = t

    if wandb:
        wandb.log({"F1 Best Model": f1_score(best_y_trues, best_predictions)})

    # this massive amount of return data is kind of a bummer - but this kind of grew as requirements grew throughout
    # the project - this needs a refactor for cleanliness, but not now.
    return {"training_loss": training_losses,
            "testing_loss": testing_losses,
            "testing_accuracy": testing_accuracies,
            "epoch": epoch,
            "best_acc": best_acc,
            "F1 Best Model": best_f1,
            "y_true": y_true_list, "y_pred": y_pred_list,
            "best_model": best_model, "best_epoch": best_epoch, 'best_model_y_trues':best_y_trues,
            "best_model_y_preds": best_predictions, "ending_epoch": last_epoch, "best_epoch": best_epoch}


def plot_results(history_dict,
                 folder_path,
                 title,
                 ) -> None:
    """

    :param history_dict: a history dictionary that comes from the function above
    :param folder_path: a folder that you want to dump the contents into
    :param title: the title - typically the name of the model you're working with
    :return: None - matplotlib does it's thing and saves to a file
    """
    # clear out the old axes and plots - otherwise problems arise later.
    plt.cla()
    plt.clf()

    training_loss = history_dict['training_loss']
    testing_loss = history_dict['testing_loss']
    testing_accuracy = history_dict['testing_accuracy']
    epochs = history_dict['epoch']

    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, testing_loss, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"{title}:  Loss")
    plt.legend()
    plt.savefig(f'{folder_path}/{title}_loss.png')

    plt.cla()
    plt.clf()

    plt.plot(epochs, testing_accuracy, label='Testing Accuracy')
    plt.title(f"{title}:  Testing Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{folder_path}/{title}_testacc.png')
