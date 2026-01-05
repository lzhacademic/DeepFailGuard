from .mymodel import MyModel
from torchvision import models
from torch import nn


class SqueezeNet(MyModel):
    def __init__(self, num_classes):
        super().__init__('squeezenet', num_classes)

    def build_model(self):
        # Load the pretrained model from pytorch
        model = models.squeezenet1_0(pretrained=False)
        # Change the last layer to match the number of classes
        model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
        return model
    
    def change_model_structure(self):
        model = self.model_structure
        del model.features[4]  # 删除第 4 个 Fire 模块
        del model.features[8]  # 删除第 8 个 Fire 模块
        return model


