import torch
import os
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from collections.abc import Iterable
from sklearn.utils import shuffle
from collections import Counter
from typing import Tuple, Dict
from torchvision.datasets import Flowers102
from torchvision.transforms import ToTensor, Compose, Lambda
from torchvision.models import ViT_B_16_Weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_labels(dataset: torchvision.datasets) -> None:
    """
    Checks whether all of the possible classes are represented in a given subset.
    """
    check = Counter([str(label) for _, label in dataset])
    all_labels = all([False if str(i) not in check.keys() else True for i in range(0, 102)])
    if all_labels:
      print("There's at least 1 instance of each class")
    else:
      print("At least one class is not represented in this set")


def save_model(model, dir: str, model_name: str) -> None:
    """Saves a model to a given directory, eg. dir = "/models/new_model.pt".
    Useful for training models with freezed and unfreezed params"""

    # curr_dir = os.getcwd()
    path = f"/{dir}/"
    # path_to_model = path + f"{model_name}.pt"
    exists = os.path.exists(path)

    if not exists:
      # Create a new directory because it does not exist 
        os.makedirs(path)

    # Save the model in a given directory
    torch.save(model.state_dict(), model_name)# path_to_model)


def plot_img(img, preprocess, title=None) -> None:
    """Plots an image from the dataset after preprocessing"""
    img = np.transpose(img.numpy(), (1, 2, 0))
    mean = np.array(preprocess.mean)
    std = np.array(preprocess.std)

    img = std*img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(f"Class: {int(title)}")
    plt.show()


def load_and_transform(f: float = 0.8, weights = ViT_B_16_Weights.IMAGENET1K_V1) -> Dict:
    """Load, transform and split Flowers102 dataset into train, valid and test datasets.
    :param f: f fraction of the training dataset not allocated for a validation dataset;
    :param weights: pretrained model weights;
    :return: a dictionary with train, valid and test datasets"""

    preprocess = weights.transforms()
    transform = Compose([ToTensor(), Lambda(lambda x: preprocess(x))])

    train_dataset = Flowers102(root='.', 
                            split='train',
                            download=True,
                            transform=transform)

    test_dataset = Flowers102(root='.', 
                          split='test',
                          download=True,
                          transform=transform)

    n_train = int(f*len(train_dataset))
    n_valid = len(train_dataset) - n_train

    train_dataset, valid_dataset = random_split(train_dataset, (n_train, n_valid))
    print("Sizes of datasets: ", len(train_dataset), len(valid_dataset), len(test_dataset))

    print("Checking training dataset: ")
    check_labels(train_dataset)
    print("Checking validation dataset: ")
    check_labels(valid_dataset)
    return {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}


def unfreeze_params(model, unfreeze_params: bool=False, all: bool=False) -> None:
    """Freezes and unfreezes model parameters, useful during model training.
    :param unfreeze_params: if True, model parameters should be unfreezed
    :param all: if all parameters should be affected"""

    if unfreeze_params and all:
        for param in model.parameters():
            param.requires_grad = True

    else:
        N = len([p for p in model.parameters()])
        for idx, param in enumerate(model.parameters()):
            if idx == N-2 or idx == N-1:
                param.requires_grad = not unfreeze_params
            else:
                param.requires_grad = unfreeze_params


def valid(model, loader) -> float:
    """Validates model performance on a given loader, returns its accuracy"""
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        # initialize the number of correct predictions
        correct: int = 0 
        N: int = 0

        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            N += y.shape[0]

            # pass through the network
            output: torch.Tensor = model(x)

            # update the number of correctly predicted examples
            correct += sum([torch.argmax(output[k]) == y[k] for k in range(output.shape[0])])

    return correct / N


def run_epoch(model, optimizer, criterion, loader, optimizer2=None) -> None:
    """Runs one epoch of model training.
    :param model: ViT model to be trained,
    :param optimizer: optimizer to be used during training,
    :param criterion: criterion of comparison - predictions vs. labels,
    :param loader: loaded dataset via DataLoader,
    :param optimizer2: second optimizer used for unfreezed pretrained weights;
    """
    
    model.train()
    N: int = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        N += y.shape[0]

        #don't accumulate gradients
        optimizer.zero_grad()
        if optimizer2:
            optimizer2.zero_grad()
        output: torch.Tensor = model(x)

        loss: torch.Tensor = criterion(output, target=y)
        #backwards pass through the network
        loss.backward()

        #apply gradients
        optimizer.step()
        if optimizer2:
            optimizer2.step()


def train_with_params(params: dict, criterion,  datasets: dict, ViT_path: str, 
                    unfreezed: bool = False, at_beginning: bool=False, lr2: float = 1e-4) -> Tuple:
    """
    :param params: a dictionary containing parameters 'batch_size', 'lr', etc.;
    :param criterion: criterion to be used during training epochs;
    :param datasets: a dictionary containting datasets for training, validation and testing
    :param :unfreezed: train the model during the last epoch with unfreezed all wieghts;
    :param at_beginning: if True, parameters will be unfreezed for the first epoch of training;
    :param lr2: learning rate used for optimizer2, for tuning the pretrained weights;
    :return: model validation accuracy, trained ViT model
    """
    train_dataset, valid_dataset = datasets["train"], datasets["valid"]
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False)

    # test_model = BasicViT().to(device)  # for random weight initialization
    test_model = torch.load(ViT_path)  # use the same baseline model for each experiment
    test_model = test_model.to(device)

    optimizer = torch.optim.Adam([p for p in test_model.parameters() if p.requires_grad], lr=params['lr'])
    optimizer2 = None

    if unfreezed:
        unfreeze_params(test_model, unfreeze_params=True, all=False)
        # print("After switching grads ON: ",len([p for p in test_model.parameters() if p.requires_grad]))
        optimizer2 = torch.optim.Adam([p for p in test_model.parameters() if p.requires_grad], lr=lr2)
        unfreeze_params(test_model, unfreeze_params=False, all=False)

    for epoch in range(params["epochs_num"]):
        if at_beginning and epoch == 0 and unfreezed:
            print("Training with unfreezed params, first epoch")
            unfreeze_params(test_model, unfreeze_params=True, all=True) # unfreeze all params
            run_epoch(test_model, optimizer, criterion, train_loader, optimizer2=optimizer2)
            unfreeze_params(test_model, unfreeze_params=False) # freeze params back

        elif not at_beginning and epoch == params["epochs_num"]-1 and unfreezed:
            print("Training with unfreezed params, last epoch")
            unfreeze_params(test_model, unfreeze_params=True, all=True)
            run_epoch(test_model, optimizer, criterion, train_loader, optimizer2=optimizer2)
            # unfreeze_params(test_model, unfreeze_params=False)
        else:
            if epoch == 0:
                print("Model with parameters:", params)
            print(f"Training with freezed params, epoch = {epoch+1}")
            run_epoch(test_model, optimizer, criterion, train_loader)

    model_valid_acc = valid(test_model, valid_loader)
    torch.cuda.empty_cache()

    return model_valid_acc, test_model


def make_params_grid(param_grid, max_num_sets=None, randomize=True):
    """Returns a Grid of parameters for loading data and tarining model"""
    to_list = lambda x: [x] if not isinstance(x, Iterable) else x

    params = {k: to_list(v) for k, v in param_grid.items()}
    if randomize:
        grid = shuffle(ParameterGrid(params))
        return grid[:max_num_sets]

    return ParameterGrid(params)


def find_best_params(param_grid, max_num_sets, criterion, datasets, ViT_path: str,
                    unfreezed=False, at_beginning=False,
                    ViT_best_path: str = "BEST_PARAMS_MODEL.pt") -> Dict:
    """Resturns a dictionary with the best hyperparameters for model training and data loading"""

    best_params = {}
    best_valid_acc = 0.0

    param_grid = make_params_grid(param_grid, max_num_sets, randomize=True)

    for i, params in enumerate(param_grid):
        model_valid_acc, trained_model = train_with_params(params=params, criterion=criterion, datasets=datasets, \
                                                          unfreezed=unfreezed, at_beginning=at_beginning, ViT_path=ViT_path)
        if max_num_sets > 1:
            print(f'Model: {i} trained, valid accuracy: {model_valid_acc:.4f}')
        else:
            print(f'Model trained, valid accuracy: {model_valid_acc:.4f}')

        if model_valid_acc > best_valid_acc:
            best_valid_acc = model_valid_acc
            best_params = params
            torch.save(trained_model, ViT_best_path)
        torch.cuda.empty_cache()

    if max_num_sets > 1:
        print(f'Best params: {best_params}, best validation accuracy: {best_valid_acc:.4f}')

    test_loader = DataLoader(datasets["test"], batch_size=best_params['batch_size'], shuffle=False)
    best_model = torch.load(ViT_best_path)
    print(f'Test accuracy: {valid(best_model, test_loader):.4f}' )

    return best_params