import torchvision.datasets as datasets
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

"""
This is a simple wrap of the CIFAR datasets.
TorchVision will automatically download the dataset to the given directory, but the directory must be created first.
If TorchVision fails to download, you may download the dataset from the address in the console logs.
"""


def get_dataset_100(dir, train=True, **kwargs):
    dataset = datasets.CIFAR100(
        root=dir,
        train=train,
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)
    return dataset


def get_dataset_10(dir, train=True, **kwargs):
    dataset = datasets.CIFAR10(
        root=dir,
        train=train,
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)
    return dataset
