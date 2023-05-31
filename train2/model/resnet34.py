import torchvision.models as models
import torch.nn as nn

class ResNet34(nn.Module):
    def __init__(self, num_classes, pretrained=True, only_last_layer=False):
        super(ResNet34, self).__init__()

        self.resnet34 = models.resnet34(pretrained=pretrained)

        if only_last_layer:
            for param in self.resnet34.parameters():
                param.requires_grad = False

        self.resnet34.fc = nn.Linear(self.resnet34.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet34(x)
