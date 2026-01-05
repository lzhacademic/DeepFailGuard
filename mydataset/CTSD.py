import os
import torch
from .mydata import MyData
from torchvision import transforms

class CTSDData(MyData):
    def __init__(self):
        super().__init__('CTSD')

    def load_data(self):
        CTSD_path = os.path.abspath('mydataset/data/CTSD')
        x_train = torch.load(CTSD_path + '/x_train.pth', weights_only=False)
        y_train = torch.load(CTSD_path + '/y_train.pth', weights_only=False) 
        x_test = torch.load(CTSD_path + '/x_test.pth',   weights_only=False)
        y_test = torch.load(CTSD_path + '/y_test.pth',   weights_only=False)
        return x_train, y_train, x_test, y_test

    def transform_method(self):
        return transforms.Compose([])


if __name__ == '__main__':
    CTSD = CTSDData()