from model import MLPDigitRecogniserModel, CNNDigitRecogniserModel
from dataset import load_mnist
from predict import predict
from utils import select_device

from torch import nn


def main():
    model1 = MLPDigitRecogniserModel()
    model2 = CNNDigitRecogniserModel()

    dataloader = load_mnist()

    device = select_device()

    batch, labels = next(iter(dataloader))

    print("MLP output logits:")
    outputs1 = model1(batch)
    print(outputs1)
    print()

    print("CNN output logits:")
    outputs2 = model2(batch)
    print(outputs2)
    print()

    loss_function = nn.CrossEntropyLoss()
    loss1 = loss_function(outputs1, labels)
    loss2 = loss_function(outputs2, labels)
    print(f"loss type: {type(loss1)}")
    print(f"Loss 1: {loss1}, Loss 2: {loss2}")
    print()

    print(f"Batch shape: {batch.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Outputs 1 shape: {outputs1.shape}")
    print(f"Outputs 2 shape: {outputs2.shape}")
    print()

    print("State dict 1:")
    state_dict = model1.state_dict()
    print("Keys, types and shapes of value:")
    for key in state_dict:
        print(f"{key} | {type(state_dict[key])} | {state_dict[key].shape}")
    print()

    print("State dict 2:")
    state_dict = model2.state_dict()
    print("Keys, types and shapes of value:")
    for key in state_dict:
        print(f"{key} | {type(state_dict[key])} | {state_dict[key].shape}")
    print()

    print("Predictions: ")
    print(f"Model 1 pred: {predict(model=model1, image_tensor=batch, device=device)}")
    print(f"Model 2 pred: {predict(model=model2, image_tensor=batch, device=device)}")
    print(f"Correct answer: {labels}")


if __name__ == "__main__":
    main()
