from __future__ import annotations

import torch

from model import CNNDigitRecogniserModel, MLPDigitRecogniserModel


def predict(
    model: CNNDigitRecogniserModel | MLPDigitRecogniserModel,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> list[int]:
    """Predict the MNIST digit represented by one image tensor.

    Args:
        model: A trained MLP or CNN digit recogniser.
        image_tensor: One grayscale 28x28 image with shape ``(1, 28, 28)``.
        device: Device on which to run inference.

    Returns:
        The list of predicted digits, from 0 to 9.

    Raises:
        ValueError: If ``image_tensor`` does not represent a valid MNIST image
    """
    if image_tensor.ndim == 3:
        if tuple(image_tensor.shape) != (1, 28, 28):
            raise ValueError(
                "image_tensor must have shape (1, 28, 28) for one MNIST image; "
                f"received {tuple(image_tensor.shape)}."
            )
        image_tensor = image_tensor.unsqueeze(0)
    elif image_tensor.ndim == 4:
        if image_tensor.shape[0] == 0 or tuple(image_tensor.shape[1:]) != (1, 28, 28):
            raise ValueError(
                "image_tensor must have shape (B, 1, 28, 28) for one MNIST image; "
                f"received {tuple(image_tensor.shape)}."
            )
    else:
        raise ValueError(
            "image_tensor must have shape (1, 28, 28) or (B, 1, 28, 28); "
            f"received {tuple(image_tensor.shape)}."
        )

    model = model.to(device)
    model.eval()
    image_tensor = image_tensor.to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        logits = model(image_tensor)
        return logits.argmax(dim=1).tolist()
