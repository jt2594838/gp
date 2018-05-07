import torchvision.datasets as datasets
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def get_dataset_100(dir, train=True):
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


def get_dataset_10(dir, train=True):
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
