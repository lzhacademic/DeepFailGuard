from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
import numpy as np


class MyData(metaclass=ABCMeta):
    def __init__(self, dataset_name):
        self.bug_id = None
        self.dataset_name = dataset_name.upper()
        self.transform = self.transform_method()
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.x_train, self.y_train, self.x_test, self.y_test = np.array(self.x_train), np.array(self.y_train), np.array(self.x_test), np.array(self.y_test)
        self.x_train, self.x_val, self.y_train, self.y_val = self.split_data()
        self.num_classes = self.cal_num_classes()
        self.train_loader, self.val_loader, self.test_loader = None, None, None

    def __str__(self):
        return f"Dataset: {self.dataset_name}, bugID: {self.bug_id}, num_classes: {self.num_classes}\nTrain: {self.x_train.shape}, {self.y_train.shape}\nVal: {self.x_val.shape}, {self.y_val.shape}\nTest: {self.x_test.shape}, {self.y_test.shape}"

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def transform_method(self):
        pass

    def split_data(self):
        x_train, x_val, y_train, y_val = train_test_split(self.x_train, self.y_train, test_size=0.2, random_state=42, shuffle=True)
        return x_train, x_val, y_train, y_val

    def cal_num_classes(self):
        return len(set(self.y_train.tolist() + self.y_val.tolist() + self.y_test.tolist()))
