from .mymodel import MyModel
from torchvision import models
from torch import nn


class AlexNet(MyModel):
    def __init__(self, num_classes):
        super().__init__('alexnet', num_classes)

    def build_model(self):
        # Load the pretrained model from pytorch
        model = models.alexnet(pretrained=False)
        # Change the last layer to match the number of classes
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, self.num_classes)
        return model

    def change_model_structure(self):
        model = self.model_structure
        del model.features[11]
        del model.features[10]
        del model.classifier[5]
        del model.classifier[4]
        del model.classifier[3]
        del model.classifier[0]
        return model
