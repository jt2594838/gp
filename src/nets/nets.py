import torchvision.models as models
import torch.nn as nn


def get_ResNet(num_classes):
    model = models.resnet101(pretrained=True)
    classifier_in_features = model.fc.in_features
    model.fc = nn.Linear(classifier_in_features, num_classes)
    return model


def get_VGG16(num_classes):
    model = models.vgg16_bn(pretrained=True)
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
    "ResNet": get_ResNet
}

if __name__ == '__main__':
    model = net_factory['ResNet'](10)
    print(model)
