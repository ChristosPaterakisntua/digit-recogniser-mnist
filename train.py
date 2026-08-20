from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from pathlib import Path

from model import CNNDigitRecogniserModel, MLPDigitRecogniserModel

EPOCHS = 50
PATIENCE = 7
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
CHECKPOINT_PATH = "saved_models/checkpoint.pt"
CHECKPOINT_FREQUENCY = 1


def loss_plotter(loss_list: list[float]):
    plt.plot(range(1, len(loss_list) + 1), loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per epoch")
    plt.show()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    epochs_without_improvement: int,
    train_loss_list: list[float],
    path: str = CHECKPOINT_PATH,
) -> None:
    """
    Saves a training checkpoint as a python dict object inside a .pt file
    with the following information:
    - epoch : `int`
        the last epoch
    - model_state_dict : `dict[str, Any]`
    - optimizer_state_dict : `dict[str, Any]`
    - best_loss : `float`
    - epochs_without_improvement : `float`
    - train_loss_list : `list[float]`

    Args
    --------
    model : nn.Module,
    optimizer : torch.optim.Optimizer,
    epoch : int,
    best_loss : float,
    epochs_without_improvement : int,
    train_loss_list : list[float],
    path : str = :data:`CHECKPOINT_PATH`,
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "train_loss_list": train_loss_list,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> tuple[int, float, int, list[float]]:
    """
    Loads a saved checkpoint

    Args
    ------
    model : nn.Module,
        model gets the saved model state
    optimizer : torch.optim.Optimizer,
        optimizer gets the saved optimizer state
    path : str,
        path pointing to saved checkpoint
    device : torch.device,

    Returns
    -------
    (start epoch, best loss, epochs without improvement, train loss list) : tuple[int, float, int, list[float]]
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        checkpoint["epoch"] + 1,
        checkpoint["best_loss"],
        checkpoint["epochs_without_improvement"],
        checkpoint["train_loss_list"],
    )


def train(
    model: CNNDigitRecogniserModel | MLPDigitRecogniserModel,
    data_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    plot_loss: bool = True,
    model_id: str = "",
    checkpoint_frequency: int = CHECKPOINT_FREQUENCY,
    resume: bool = False,
    checkpoint_path: str | None = None,
) -> None:
    """
    Trains given model and saves it as .pt file

    Args
    -------
    model : :class:`CNNDigitRecogniserModel` | :class:`MLPDigitRecogniserModel`

    data_loader : DataLoader

    device : torch.device

    epochs : int = :data:`EPOCHS`

    learning_rate : float = :data:`LEARNING_RATE`

    weight_decay : float = :data:`WEIGHT_DECAY`

    plot_loss : bool = `True`
        If true it shows a plot with train loss vs epoch number

    model_id : str = `""`,
        The model is saved as `model_<model_id>`

    checkpoint_frequency : int = :data:`CHECKPOINT_FREQUENCY`
        Saves a checkpoint every `checkpoint_frequency` epoch(s)

    resume : bool
        If `True` the training resumes from a saved checkpoint

    checkpoint_path : str | None = None
        path pointing to saved checkpoints used for loading and saving checkpoints
    """
    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_loss_list: list[float] = []
    epochs_without_improvement = 0
    best_loss = float("inf")
    min_delta = 0.001
    start_epoch = 0

    model = model.to(device)

    if resume:
        if checkpoint_path is None:
            raise ValueError("When resume is True checkpoint_path must be provided")
        (start_epoch, best_loss, epochs_without_improvement, train_loss_list) = (
            load_checkpoint(
                model=model,
                optimizer=optimizer,
                path=checkpoint_path,
                device=device,
            )
        )

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = 0
        model.train()
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_list.append(train_loss / len(data_loader))

        current_loss = train_loss_list[-1]

        if best_loss - current_loss > min_delta:
            best_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"Training loss: {train_loss_list[-1]}")

        if checkpoint_path is not None and ((epoch + 1) % checkpoint_frequency == 0):
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_loss=best_loss,
                epochs_without_improvements=epochs_without_improvement,
                train_loss_list=train_loss_list,
                path=checkpoint_path,
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch: {epoch + 1}")
            break

    if plot_loss:
        loss_plotter(loss_list=train_loss_list)

    path = f"saved_models/model_{model_id}.pt"
    Path(path).parent.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), path)
    print(f"Final model saved: {path}")
