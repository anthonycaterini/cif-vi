import torch

from .supervised_dataset import SupervisedDataset

DUMMY_DATASET_SIZE = 1000


def get_dummy_datasets():
    dummy_data = torch.zeros((DUMMY_DATASET_SIZE, 1), dtype=torch.get_default_dtype())
    train_dset = SupervisedDataset("dummy", "train", dummy_data)
    valid_dset = SupervisedDataset("dummy", "valid", dummy_data)
    test_dset = SupervisedDataset("dummy", "test", dummy_data)
    return train_dset, valid_dset, test_dset
