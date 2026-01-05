import os
import torch
from .mydata import MyData
from torchvision import transforms

class GTSRBData(MyData):
    def __init__(self):
        super().__init__('GTSRB')

    def load_data(self):
        GTSRB_path = os.path.abspath('mydataset/data/GTSRB')
        x_train = torch.load(GTSRB_path + '/x_train.pth', weights_only=False)
        y_train = torch.load(GTSRB_path + '/y_train.pth', weights_only=False) 
        x_test = torch.load(GTSRB_path + '/x_test.pth',   weights_only=False)
        y_test = torch.load(GTSRB_path + '/y_test.pth',   weights_only=False)
        return x_train, y_train, x_test, y_test

    def transform_method(self):
        return transforms.Compose([])


if __name__ == '__main__':
    GTSRB = GTSRBData()