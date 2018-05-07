import torchvision.datasets as datasets
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def get_dataset(dir):
    dataset = datasets.ImageFolder(
        dir,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    return dataset
