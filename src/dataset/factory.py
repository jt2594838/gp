from dataset import CIFARDataset, H5Dataset

dataset_factory = {
    'CIFAR_100': CIFARDataset.get_dataset_100,
    'CIFAR_10': CIFARDataset.get_dataset_10,
    'H5': H5Dataset.get_anzhen_dataset,
}