import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.autograd import Variable
from nets import Resnet


class ConvDeconvV2(nn.Module):
    def __init__(self, in_channels):
        super(ConvDeconvV2, self).__init__()
        self.encoder = Resnet.resnet50(pretrained=False, in_channels=in_channels)
        self.encoded_features = 1024
        self.decoder = nn.Sequential(
            nn.Conv2d(self.encoded_features, 256, kernel_size=1, stride=1, padding=0),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1,
                      bias=False),
        )
        self.output = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output(x)
        return x


if __name__ == '__main__':
    net = ConvDeconvV2(3)
    input = torch.rand((1, 3, 224, 224))
    output = net(Variable(input))
    print(input, output)
