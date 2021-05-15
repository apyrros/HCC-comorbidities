import torch
import torchvision.models as models
import torch.nn as nn

class ResNet34(nn.Module):
    def __init__(self, num_classes, pretrained=True, only_last_layer=False):
        '''
        ResNet34 init function

        @param num_classes : number of output classes
        @param pretrained : use pretrained resnet34 model?
        @param only_last_layer : fine-tune only the last layer?
        '''
        super(ResNet34, self).__init__()

        self.resnet34 = models.resnet34(pretrained=pretrained)

        if only_last_layer:
            for param in self.resnet34.parameters():
                param.requires_grad = False

        self.resnet34.fc = nn.Linear(self.resnet34.fc.in_features, num_classes)

    def forward(self, x):
        '''
        ResNet34 forward pass

        @param x : input tensor (B x C x H x W)
        @return pred : model predictions (B x num_classes)
        '''
        return self.resnet34(x)
