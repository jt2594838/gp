from dataset import CIFARDataset, ImageNetDataset, H5Dataset

dataset_factory = {
    'CIFAR_100': CIFARDataset.get_dataset_100,
    'CIFAR_10': CIFARDataset.get_dataset_10,
    'ImageNet': ImageNetDataset.get_dataset,
    'anzhen': H5Dataset.get_anzhen_dataset,
}