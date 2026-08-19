from PIL import Image, ImageOps
import torch
from torchvision import transforms


def load_raw_image(path: str) -> torch.Tensor:
    """
    Loads a raw image and returns a Tensor of shape [1,28,28]
    ([channels, height, width]). The image is converted to grayscale and
    resized to 28x28 px. The purpose of this function is to produce tensors
    that match mnist dataset tensors.
    """
    path = path.strip().strip('"')
    with Image.open(path) as img:
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img)
        img = img.resize(size=(28, 28))
        # remove this
        img.show()

        transform = transforms.ToTensor()
        img_tensor = transform(img)
    print(img_tensor.dtype)
    print(img_tensor.shape)
    print(img_tensor.min())
    print(img_tensor.max())
    return img_tensor


def main():
    path = input("Path: ")
    print(load_raw_image(path))


if __name__ == "__main__":
    main()
