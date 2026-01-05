import os
import torch
from .mydata import MyData
from torchvision import transforms

class ATSData(MyData):
    def __init__(self):
        super().__init__('ATS')

    def load_data(self):
        ATS_path = os.path.abspath('mydataset/data/ATS')
        x_train = torch.load(ATS_path + '/x_train.pth', weights_only=False)
        y_train = torch.load(ATS_path + '/y_train.pth', weights_only=False) 
        x_test = torch.load(ATS_path + '/x_test.pth',   weights_only=False)
        y_test = torch.load(ATS_path + '/y_test.pth',   weights_only=False)
        return x_train, y_train, x_test, y_test

    def transform_method(self):
        return transforms.Compose([])


if __name__ == '__main__':
    ats = ATSData()