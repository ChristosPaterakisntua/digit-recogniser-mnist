from model import MLPDigitRecogniserModel, CNNDigitRecogniserModel
from dataset import load_mnist

from torch import nn


def main():
    model1 = MLPDigitRecogniserModel()
    model2 = CNNDigitRecogniserModel()

    dataloader = load_mnist()

    batch, labels = next(iter(dataloader))

    print("MLP output logits:")
    outputs1 = model1(batch)
    print(outputs1)

    print("CNN output logits:")
    outputs2 = model2(batch)
    print(outputs2)

    loss_function = nn.CrossEntropyLoss()
    loss1 = loss_function(outputs1, labels)
    loss2 = loss_function(outputs2, labels)
    print(f"loss type {type(loss1)}")
    print(f"Loss 1: {loss1}, Loss 2: {loss2}")

    print(f"Batch shape: {batch.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Outputs 1 shape: {outputs1.shape}")
    print(f"Outputs 2 shape: {outputs2.shape}")


if __name__ == "__main__":
    main()
