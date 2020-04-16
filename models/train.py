from random import shuffle, seed
import os
import pathlib
from time import time

import torch
import numpy as np
from torch import nn, optim, no_grad

from models.neural_nets import (MNIST_MLP, MNIST_LeNet, FashionMNIST_MLP, FashionMNIST_LeNet,
    SVHN_LeNet, SVHN_LeNet_BandW, CIFAR_LeNet)
from models.resnet import ResNet18
from models.datasets import Dataset
from utils.tools import get_logger, device, rootpath
torch.set_default_tensor_type(torch.DoubleTensor)

logger = get_logger("Train")

if not os.path.exists(f"{rootpath}/trained_models"):
        os.mkdir(f"{rootpath}/trained_models")

##### Paramaters of training
# Epochs
# (Pruning ratio)
# (Adversarial training)

def get_model(architecture):
    if architecture == "MNIST_MLP":
        model = MNIST_MLP()
    elif architecture == "MNIST_LeNet":
        model = MNIST_LeNet()
    elif architecture == "FashionMNIST_MLP":
        model = FashionMNIST_MLP()
    elif architecture == "FashionMNIST_LeNet":
        model = FashionMNIST_LeNet()
    elif architecture == "SVHN_LeNet":
        model = SVHN_LeNet()
    elif architecture == "SVHN_LeNet_BandW":
        model = SVHN_LeNet_BandW()
    elif architecture == "CIFAR_LeNet":
        model = CIFAR_LeNet()
    elif architecture == "ResNet18":
        model = ResNet18()
    return model

def compute_accuracy(model, loader):
    model.eval()
    correct = 0
    with no_grad():
        for data, target in loader:
            data = data.double()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct / len(loader.dataset)
    return acc

def go_training(model, x, y, epoch, optimizer, loss_func):
    x = x.double()
    y = y.to(device)
    optimizer.zero_grad()

    # Usual training
    y_pred = model(x)
    loss = loss_func(y_pred, y)
    loss.backward()
    optimizer.step()

def eval_training_val(model, dataset, x_val, y_val, epoch, optimizer, scheduler, loss_func):
    y_val = y_val.to(device)
    x_val = x_val.double()
    y_val_pred = model(x_val)
    val_loss = loss_func(y_val_pred, y_val)
    logger.info(f"Validation loss = {np.around(val_loss.item(), decimals=4)}")
    scheduler.step(val_loss)
    if epoch % 10 == 0:
        logger.info(f"Val acc = {compute_accuracy(model, dataset.val_loader)}")
    return val_loss.item()


def training_nn(dataset, architecture, epochs, loss_func, pruning, adv_training, model_filename):

    # Init model and saving
    model = get_model(architecture)
    torch.save(model.state_dict(), f"{model_filename}_initial.pt")
    logger.info(f"Saved initial model in {model_filename}_initial.pt")

    # Preparing parameters for training
    if architecture in ["MNIST_LeNet", "FashionMNIST_LeNet"]:
        lr = 0.001
        patience = 20
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, verbose=True, factor=0.5)
    elif architecture in ["SVHN_LeNet", "SVHN_LeNet_BandW", "CIFAR_LeNet"]:
        lr = 0.0008
        patience = 40
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, verbose=True, factor=0.5)
    elif architecture in ["MNIST_MLP", "FashionMNIST_MLP"]:
        lr = 0.1
        patience = 5
        optimizer = optim.SGD(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, verbose=True, factor=0.5)

    # Initialization
    loss_history = []
    t = time()

    # Training
    for epoch in range(epochs):
        logger.info(
            f"Starting epoch {epoch} ({time()-t} secs), lr = {[param['lr'] for param in optimizer.param_groups]}"
        )
        t = time()
        model.train()

        for x_batch, y_batch in dataset.train_loader:
            go_training(model, x_batch, y_batch, epoch,
                optimizer, loss_func)

        model.eval()
        for x_val, y_val in dataset.val_loader:
            loss_val = eval_training_val(model, dataset, x_val, y_val, epoch, optimizer, scheduler, loss_func)
            loss_history.append(loss_val)

    return model, loss_history
            

def get_my_model(dataset_name, architecture, epochs, loss_func, pruning, adv_training):

    dataset = Dataset.get_or_create(name=dataset_name)

    if pruning > 0.0:
        pruning_message = f"_{pruning}_pruning"
    else:
        pruning_message = ""
    if adv_training:
        adv_training_message = f"_pgd_adv_train"
    else:
        adv_training_message = f""

    model_filename = (
                f"{rootpath}/trained_models/{dataset.name}_"
                f"{architecture}_"
                f"{epochs}_epochs"
                f"{pruning_message}"
                f"{adv_training_message}"
            )
    logger.info(f"Filename = {model_filename} \n")

    try:
        model = get_model(architecture)
        model.load_state_dict(torch.load(f"{model_filename}.pt"))
        logger.info(f"Loaded successfully model from {model_filename}")
    except FileNotFoundError:
        logger.info(f"Unable to find model in {model_filename}... Retraining it...")
        model, loss_history = training_nn(
            dataset,
            architecture,
            epochs,
            loss_func,
            pruning,
            adv_training,
            model_filename
            )
        torch.save(model.state_dict(), f"{model_filename}.pt")
        logger.info(f"Saved trained model in {model_filename}.pt")

    logger.info(f"Checking test accuracy = {compute_accuracy(model, dataset.test_loader)}")

    return model

