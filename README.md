# MNIST Digit Recogniser

An educational PyTorch project for recognising handwritten digits from the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. Its main purpose is to compare a **Multi-Layer Perceptron (MLP)** with a **Convolutional Neural Network (CNN)** and demonstrate why CNNs are generally better suited to image-recognition tasks.

## Learning goals

- Load, normalise, and batch the MNIST dataset.
- Train and evaluate neural-network classifiers with PyTorch.
- Compare a fully connected MLP against a CNN.
- Observe how convolutions and pooling make use of the spatial structure of images.
- Save training checkpoints and trained-model weights.

## Why compare MLP and CNN?

MNIST images are 28 x 28 grayscale images. An MLP flattens each image into a single vector, so neighbouring pixels are no longer explicitly related. A CNN keeps the two-dimensional image layout and learns local patterns such as edges, curves, and digit strokes through convolutional filters.

For this reason, the CNN is expected to achieve better image-classification performance and be more appropriate for handwritten-digit recognition. The comparison is intended as a learning exercise, not as a benchmark claim.

## Models

### MLP

`MLPDigitRecogniserModel` uses the following classifier:

`Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 10)`

### CNN

`CNNDigitRecogniserModel` uses three convolutional stages followed by fully connected layers:

`Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> Flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear(10)`

## Core Requirements

- Python 3
- PyTorch
- torchvision
- matplotlib
- Pillow
- Jupyter Notebook (optional, for `main.ipynb`)

Install the Python packages in a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> Install the PyTorch build appropriate for your CPU or GPU from the [official PyTorch installation page](https://pytorch.org/get-started/locally/) if needed.

## Run the project

### Notebook

Open and run [main.ipynb](main.ipynb). The MNIST dataset is downloaded automatically into the `data/` directory on first use.

The script trains a CNN, evaluates it on the MNIST test set, and can optionally predict digits from local image files.

## Comparing CNN and MLP

Use distinct model IDs (for example, `CNN_1` and `MLP_1`), train both models with the same parameters, and compare the printed test accuracies. The CNN should normally perform better because it preserves and learns from image structure.

## Saved files

During training, the project writes checkpoints and final weights to `saved_models/`:

- `checkpoint.pt` — checkpoint data used to resume training.
- `model_<model_id>.pt` — final model weights, for example `model_CNN_1.pt`.

The training configuration is defined in `train.py`, including epochs, learning rate, weight decay, early stopping patience, and checkpoint frequency.

## File Overview

| File | Description |
| --- | --- |
| `dataset.py` | MNIST dataset loading |
| `model.py` | MLP and CNN definitions |
| `train.py` | Training, early stopping, and checkpoints |
| `evaluate.py` | Test-set evaluation |
| `predict.py` | Inference helper |
| `loader.py` | Local image preprocessing |
| `main.ipynb` | Notebook workflow |
| `test.py` | Prints useful data for understanding the process |
| `utils.py` | General helful functions |
| `output.pdf` | Notebook output |
| `.gitignore` | Specifies files that shouldn't be included |
| `LICENSE` | license |
| `requirements.txt` | Versions of python libraries |
| `README.md` | Project documentation |
| `saved_models/` | includes the trained cnn model |

## License

This project is released under the [MIT License](LICENSE).
