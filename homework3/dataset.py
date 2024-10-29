import numpy as np
from torchvision.datasets import MNIST
def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
    transform=lambda x: np.array(x).flatten(),
    download=True,
    train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return mnist_data, mnist_labels
train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)
