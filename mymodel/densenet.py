from .mymodel import MyModel
from torchvision import models
from torch import nn, flatten


class NewDenseNetModel(nn.Module):
    # bug1 for DenseNet. suboptimal network structure
    def __init__(self, model):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.densenet = model
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.densenet.features.conv0(x)
        x = self.densenet.features.norm0(x)
        x = self.densenet.features.relu0(x)
        x = self.densenet.features.pool0(x)

        x = self.densenet.features.denseblock1(x)
        x = self.densenet.features.transition1(x)
        x = self.densenet.features.denseblock2(x)
        x = self.densenet.features.transition2(x)
        x = self.densenet.features.denseblock3(x)
        x = self.densenet.features.transition3(x)
        x = self.densenet.features.denseblock4(x)
        x = self.densenet.features.norm5(x)
        x = self.avg_pool(x)
        x = flatten(x, 1)
        x = self.densenet.classifier(x)

        return x


class DenseNet(MyModel):
    def __init__(self, num_classes):
        super().__init__(model_name='densenet', num_classes=num_classes)

    def build_model(self):
        # Load the pretrained model from pytorch
        model = models.densenet121(pretrained=False)
        # Change the last layer to match the number of classes
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, self.num_classes)
        return model

    def change_model_structure(self):
        model = self.model_structure
        model = NewDenseNetModel(model)
        return model
