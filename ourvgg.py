#implementation of the cnn we change the last layer from 1000 to 100.
import torch
import torch.nn as nn
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG, self).__init__()
        self.features = models.vgg19().features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    #Kaiming initialization
    def initialize_weights(self):
        for m in self.modules():
            if(isinstance(m, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight)