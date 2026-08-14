from __future__ import annotations

import torch
from torch import nn


IMAGE_WIDTH, IMAGE_HEIGHT = 28, 28


class MLPDigitRecogniserModel(nn.Module):
    """
    Classification Model

    Input
    -------
    images tensor:
        Shape: (batch_size, channels, height, width)
        Default shape: (:data:`dataset.DEFAULT_BATCH_SIZE`, 1, :data:`IMAGE_HEIGHT`, :data:`IMAGE_WIDTH`)
        -> (5, 1, 28, 28)

    Output
    -------
    digit_logits:
        Shape: (batch_size, 10)
        Default shape: (:data:`dataset.DEFAULT_BATCH_SIZE`, 10)

    Note
    -----
    Logits are produced by:
    flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMAGE_HEIGHT * IMAGE_WIDTH, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Shape of `image_tensor`: (batch_size, 1, :data:`IMAGE_HEIGHT`, :data:`IMAGE_WIDTH`)
        Returned shape: (batch_size, 10)
        """
        return self.model(image_tensor)  # logits


class CNNDigitRecogniserModel(nn.Module):
    """
    Classification Model

        Input
        -------
        images tensor:
            Shape: (batch_size, channels, height, width)
            Default shape: (:data:`dataset.DEFAULT_BATCH_SIZE`, 1, :data:`IMAGE_HEIGHT`, :data:`IMAGE_WIDTH`)
            -> (5, 1, 28, 28)

        Output
        -------
        digit_logits:
            Shape: (batch_size, 10)
            Default shape: (:data:`dataset.DEFAULT_BATCH_SIZE`, 10)

        Note
        -----
        Logits are produced by:
        Conv2d -> ReLU -> MaxPool2d ->
        Conv2d -> ReLU -> MaxPool2d ->
        Conv2d -> ReLU ->
        Flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            # (B, 1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            # (B, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # (Β, 32, 14, 14)
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            # (Β, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # (Β, 64, 7, 7)
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
            # (Β, 64*7*7 = 3136)
            nn.Linear(
                in_features=64 * 7 * 7,
                out_features=512,
            ),
            # (Β, 512)
            nn.ReLU(),
            nn.Linear(
                in_features=512,
                out_features=64,
            ),
            # (Β, 64)
            nn.ReLU(),
            nn.Linear(
                in_features=64,
                out_features=10,
            ),
            # (B, 10)
        )

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(img_tensor)
