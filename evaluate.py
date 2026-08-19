from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from dataset import load_mnist
from model import CNNDigitRecogniserModel, MLPDigitRecogniserModel


def load_trained_model(
    path: str, device: torch.device
) -> CNNDigitRecogniserModel | MLPDigitRecogniserModel:
    return torch.load(path.strip(), map_location=device)


def load_test_dataset() -> DataLoader:
    return load_mnist(train=False)


def evaluate(
    model: CNNDigitRecogniserModel | MLPDigitRecogniserModel,
    device: torch.device,
) -> float:
    """
    Args
    ------
    model: :class:`CNNDigitRecogniserModel` | :class:`MLPDigitRecogniserModel`
        the trained model

    Returns
    -------
    float
        test_accuracy
    """
    model = model.to(device)
    model.eval()

    test_loader = load_test_dataset()
    correct_predictions = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    return correct_predictions / total_samples
