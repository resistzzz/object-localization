

import torch
import torch.nn as nn
from torchvision import models

class AlexNet_model(nn.Module):

    def __init__(self, num_class=5, num_regress=4):

        super(AlexNet_model, self).__init__()
        alexNet = models.alexnet(pretrained=True)

        self.feature = alexNet.features
        self.classifier = alexNet.classifier
        self.classifier[6] = nn.Linear(4096, num_class, bias=True)

        self.regressor = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(9216, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_regress, bias=True)
        )

        for i in (1, 4):
            layer_c = self.classifier[i]
            layer_r = self.regressor[i]
            layer_r.weight.data.copy_(layer_c.weight.data.view(layer_r.weight.size()))
            layer_r.bias.data.copy_(layer_c.bias.data.view(layer_r.bias.size()))


    def forward(self, x):
        # x = x.view(-1, 3, 224, 224)
        x = torch.Tensor.float(x)
        x = self.feature(x)
        x = x.view(-1, 9216)

        x_classifier = self.classifier(x)
        x_regressor = self.regressor(x)

        return x_classifier, x_regressor


    def num_flat_feature(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i

        return num_features


class Vgg16_bn_model(nn.Module):

    def __init__(self, num_class=5, num_regress=4):
        super(Vgg16_bn_model, self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True)

        self.feature = vgg16_bn.features
        self.classifier = vgg16_bn.classifier
        self.classifier[6] = nn.Linear(4096, num_class, bias=True)

        self.regressor = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_regress, bias=True)
        )

        for i in (0, 3):
            layer_c = self.classifier[i]
            layer_r = self.regressor[i]
            layer_r.weight.data.copy_(layer_c.weight.data.view(layer_r.weight.size()))
            layer_r.bias.data.copy_(layer_c.bias.data.view(layer_r.bias.size()))


    def forward(self, x):

        # x = x.view(-1, 3, 224, 224)
        x = torch.Tensor.float(x)
        x = self.feature(x)
        x = x.view(-1, 7*7*512)

        x_classifier = self.classifier(x)
        x_regressor = self.regressor(x)

        return x_classifier, x_regressor


    def num_flat_feature(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i

        return num_features

    
class Resnet_model(nn.Module):

    def __init__(self, num_class=5, num_regress=4):
        super(Resnet_model, self).__init__()

        self.resnet18 = models.resnet18(pretrained=True)
        self.num_featureIn = self.resnet18.fc.in_features
        self.num_featureOut = self.resnet18.fc.out_features
        
        self.classifier = nn.Linear(self.num_featureOut, num_class, bias=True)

        self.regressor = nn.Linear(self.num_featureOut, num_regress, bias=True)
        

    def forward(self, x):
        # x = x.view(-1, 3, 224, 224)
        x = torch.Tensor.float(x)
        x = self.resnet18(x)
        
        x_classifier = self.classifier(x)
        x_regressor = self.regressor(x)
        
        return x_classifier, x_regressor


if __name__ == '__main__':

    # net = Vgg16_bn_model()
    # net = AlexNet_model()
    # net = Vgg16_bn_model()
    net = Resnet_model()
