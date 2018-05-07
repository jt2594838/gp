import torch.nn as nn

from nets import Resnet, VGG


def get_ResNet50(num_classes, pretrained=True, in_channels=3, classify=True):
    model = Resnet.resnet50(pretrained, in_channels, classify=classify)
    classifier_in_features = model.fc.in_features
    model.fc = nn.Linear(classifier_in_features, num_classes)
    return model


def get_ResNet101(num_classes, pretrained=True, in_channels=3, classify=True):
    model = Resnet.resnet101(pretrained, in_channels, classify=classify)
    classifier_in_features = model.fc.in_features
    model.fc = nn.Linear(classifier_in_features, num_classes)
    return model


def get_VGG16(num_classes, pretrained=True, in_channels=3, classify=True):
    model = VGG.vgg16_bn(pretrained, in_channels, classify=classify)
    classifier_in_features = model.classifier[0].in_features
    model.classifier = build_classifier(classifier_in_features, num_classes)
    return model


def build_classifier(input_features, output_features):
    classifier = nn.Sequential(
        nn.Linear(input_features, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, output_features)
    )
    return classifier


net_factory = {
    "VGG16": get_VGG16,
    "ResNet50": get_ResNet50,
    "ResNet101": get_ResNet101,
}

if __name__ == '__main__':
    model = net_factory['ResNet'](10)
    print(model)
