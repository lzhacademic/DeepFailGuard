from .mymodel import MyModel
from torchvision import models
from torch import nn


class ResNet(MyModel):
    def __init__(self, num_classes):
        super().__init__('resnet', num_classes)

    def build_model(self):
        # Load the pretrained model from pytorch
        model = models.resnet50(pretrained=False)
        # Change the last layer to match the number of classes
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        return model

    def change_model_structure(self):
        model = self.model_structure
        del model.layer1[2]
        del model.layer1[1]
        del model.layer2[3]
        del model.layer2[2]
        del model.layer2[1]
        del model.layer3[5]
        del model.layer3[4]
        del model.layer3[3]
        del model.layer3[2]
        del model.layer3[1]
        del model.layer4[2]
        del model.layer4[1]
        return model
