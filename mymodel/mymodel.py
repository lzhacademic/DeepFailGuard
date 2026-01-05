from abc import ABCMeta, abstractmethod


class MyModel(metaclass=ABCMeta):
    def __init__(self, model_name, num_classes):
        self.bug_id = None
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.model_structure = self.build_model()

    def __str__(self):
        return f"Model: {self.model_name}, bugID: {self.bug_id}, num_classes: {self.num_classes}"

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def change_model_structure(self):
        pass
