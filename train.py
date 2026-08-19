from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from model import CNNDigitRecogniserModel, MLPDigitRecogniserModel

EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01


def loss_plotter(loss_list: list[float]):
    plt.plot(range(1, len(loss_list) + 1), loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per epoch")
    plt.show()


def train(
    model: CNNDigitRecogniserModel | MLPDigitRecogniserModel,
    data_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    plot_loss: bool = True,
    model_id: str = "",
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
    """
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_loss_list = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = 0
        model = model.to(device)
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
        print(f"Training loss: {train_loss_list[-1]}")

    if plot_loss:
        loss_plotter(loss_list=train_loss_list)

    torch.save(model.state_dict(), f"saved_models/model_{model_id}.pt")
