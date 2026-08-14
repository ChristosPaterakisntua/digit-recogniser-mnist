from __future__ import annotations

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


DEFAULT_BATCH_SIZE = 5


def load_raw_mnist(show_details: bool = False) -> None:
    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
    )
    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
    )
    if show_details:
        print(f"{len(train_dataset)} train images")  # 60000
        print(f"{len(test_dataset)} test images")  # 10000
        image, label = train_dataset[0]
        print(f"Image type: {type(image)}")  # <class 'PIL.Image.Image'>
        print(f"Label type: {type(label)}")  # <class 'int'>
        print(f"pixels: {image.size}")  # (28, 28)
        plt.figure(figsize=(15, 3))
        for i in range(DEFAULT_BATCH_SIZE):
            image, label = train_dataset[i]
            plt.subplot(1, DEFAULT_BATCH_SIZE, i + 1)
            plt.imshow(image, cmap="gray")
            plt.title(f"Label: {label}")
            plt.axis("off")
        plt.gcf().canvas.manager.set_window_title("MNIST Dataset Preview")
        plt.show()


def load_transformed_mnist(show_details: bool = False) -> None:
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        transform=transform,
        download=True,
    )
    if show_details:
        image, label = train_dataset[0]
        print(f"Image type: {type(image)}")  # <class 'torch.Tensor'>
        print(f"Label type: {type(label)}")  # <class 'int'>
        print(f"Tensor shape: {image.shape}")  # torch.Size([1, 28, 28])


def load_mnist(
    batch_size: int = DEFAULT_BATCH_SIZE,
    train: bool = True,
) -> DataLoader:
    """
    Loads the dataset in batches

    Args
    ------
    batch_size : int
        the number of images loaded in this batch. Default: :data:`DEFAULT_BATCH_SIZE`
    train : bool
        indicates wether to use train or test dataset. Default: `True`

    Returns
    -------
    Dataloader
        Dataloader's data tensor shape: [batch_size, 1, 28, 28]
    """
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root="data",
        train=train,
        transform=transform,
        download=True,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )


if __name__ == "__main__":
    dataset = load_mnist()
    images, labels = next(iter(dataset))
    print(f"Full image tensor shape: {images.shape}")  # torch.Size([5, 1, 28, 28])
    print(f"Label tensor shape: {labels.shape}")  # torch.Size([5])
    print(
        images[0]
    )  # the grayscale image's colors are normalized (mapped to 0.0 - 1.0 where 0.0 = white, 1.0 = black)
