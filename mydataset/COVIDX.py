import os
import torch
from .mydata import MyData
from torchvision import transforms

class COVIDX9AData(MyData):
    def __init__(self):
        super().__init__('COVIDX9A')

    def load_data(self):
        COVIDX_path = os.path.abspath('mydataset/data/COVIDX9A')
        x_train = torch.load(COVIDX_path + '/x_train.pth', weights_only=False)
        y_train = torch.load(COVIDX_path + '/y_train.pth', weights_only=False)
        x_test = torch.load(COVIDX_path + '/x_test.pth',   weights_only=False)
        y_test = torch.load(COVIDX_path + '/y_test.pth',   weights_only=False)
        return x_train, y_train, x_test, y_test

    def transform_method(self):
        return transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1))])


if __name__ == '__main__':
    COVIDX9A = COVIDX9AData()