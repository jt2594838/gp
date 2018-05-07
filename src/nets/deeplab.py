import torch
import torch.nn as nn

from nets.Resnet import resnet50


class Encoder(nn.Module):

    # dilations for 2048, 14, 28, 42
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.l1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.l2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=14)
        self.upsample2 = nn.Upsample(scale_factor=3)
        self.l3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=28)
        self.upsample3 = nn.Upsample(scale_factor=4)
        self.l4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=42)
        self.output = nn.Conv2d(in_channels=out_channels * 4, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(self.upsample1(x))
        x3 = self.l3(self.upsample2(x))
        x4 = self.l4(self.upsample3(x))
        # print(x1.size(), x2.size(), x3.size(), x4.size())
        x = torch.cat((x1, x2, x3, x4), 1)
        output = self.output(x)
        return output


class Decoder(nn.Module):

    def __init__(self, feature_in_channels, code_in_channels, out_channels):
        super(Decoder, self).__init__()
        self.l1 = nn.Conv2d(in_channels=feature_in_channels, out_channels=out_channels, kernel_size=1)
        self.upsample1 = nn.Upsample(scale_factor=4)
        self.l2 = nn.Conv2d(in_channels=code_in_channels + out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)

    def forward(self, x, code):
        low_level = self.upsample1(self.l1(x))
        code = self.upsample1(code)
        cat = torch.cat((low_level, code), 1)
        output = self.l2(cat)
        output = self.upsample2(output)
        return output


class Deeplab(nn.Module):
    def __init__(self, num_classes):
        super(Deeplab, self).__init__()
        self.feature_extractor = resnet50(True)
        self.feature_channels = 2048
        self.encoder = Encoder(self.feature_channels, num_classes)
        self.decoder = Decoder(self.feature_channels, num_classes, num_classes)
        self.output = nn.Sigmoid()

    def forward(self, x):
        features = self.feature_extractor(x)
        code = self.encoder(features)
        output = self.decoder(features, code)
        output = self.output(output)
        return output


if __name__ == '__main__':
    from torch.autograd import Variable
    data = torch.zeros((1, 3, 224, 224))
    net = Deeplab(1)
    output = net(Variable(data))
    print(output)
