from .mymodel import MyModel
from torchvision import models
from torch import nn


class VGG16(MyModel):
    def __init__(self, num_classes):
        super().__init__('vgg16', num_classes)

    def build_model(self):
        # Load the pretrained model from pytorch
        model = models.vgg16(pretrained=False)
        # Change the last layer to match the number of classes
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, self.num_classes)
        return model
    
    def change_model_structure(self):
        model = self.model_structure
        del model.features[29]
        del model.features[28]
        del model.features[22]
        del model.features[21]
        del model.features[15]
        del model.features[14]

        del model.classifier[5]
        del model.classifier[4]
        del model.classifier[3]
        return model


# if __name__ == '__main__':

#     model = models.vgg16(pretrained=False)
#     print(f"model: {model}")
#     num_features = model.classifier[6].in_features
#     print(f"num_features: {num_features}")
#     print("*"*50)
#     model.classifier[6] = nn.Linear(num_features, 25)
    
#     del model.features[29]
#     del model.features[28]
#     del model.features[22]
#     del model.features[21]
#     del model.features[15]
#     del model.features[14]

#     del model.classifier[5]
#     del model.classifier[4]
#     del model.classifier[3]

#     print(f"model: {model}")
