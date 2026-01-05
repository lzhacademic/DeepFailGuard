import os
from .mydata import MyData
from torchvision.datasets import CIFAR10
from torchvision import transforms


class CIFAR10Data(MyData):
    def __init__(self):
        super().__init__('CIFAR10')

    def load_data(self):
        root = os.path.abspath('mydataset/data')
        train_dataset = CIFAR10(root=root, train=True, download=True, transform=None)
        test_dataset = CIFAR10(root=root, train=False, download=True, transform=None)
        x_train, y_train, x_test, y_test = train_dataset.data, train_dataset.targets, test_dataset.data, test_dataset.targets
        return x_train, y_train, x_test, y_test

    def transform_method(self):
        return transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])
