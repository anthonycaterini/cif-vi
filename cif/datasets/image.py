import os
import torch
import torchvision.datasets

from .supervised_dataset import SupervisedDataset


# Returns tuple of form `(images, labels)`. Both are uint8 tensors.
# `images` has shape `(nimages, nchannels, nrows, ncols)`, and has
# entries in {0, ..., 255}
def get_raw_image_tensors(dataset_name, train, data_root):
    data_dir = os.path.join(data_root, dataset_name)

    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True)
        images = torch.tensor(dataset.data).permute((0, 3, 1, 2))
        labels = torch.tensor(dataset.targets)

    elif dataset_name == "svhn":
        dataset = torchvision.datasets.SVHN(root=data_dir, split="train" if train else "test", download=True)
        images = torch.tensor(dataset.data)
        labels = torch.tensor(dataset.labels)

    elif dataset_name in ["mnist", "fashion-mnist"]:
        dataset_class = {
            "mnist": torchvision.datasets.MNIST,
            "fashion-mnist": torchvision.datasets.FashionMNIST
        }[dataset_name]
        dataset = dataset_class(root=data_dir, train=train, download=True)
        images = dataset.data.unsqueeze(1)
        labels = dataset.targets

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return images.to(torch.uint8), labels.to(torch.uint8)


def image_tensors_to_supervised_dataset(dataset_name, dataset_role, images, labels):
    images = images.to(dtype=torch.get_default_dtype())
    labels = labels.long()
    return SupervisedDataset(dataset_name, dataset_role, images, labels)


def get_train_valid_image_datasets(dataset_name, data_root, valid_fraction, add_train_hflips):
    images, labels = get_raw_image_tensors(dataset_name, train=True, data_root=data_root)

    perm = torch.randperm(images.shape[0])
    shuffled_images = images[perm]
    shuffled_labels = labels[perm]

    valid_size = int(valid_fraction * images.shape[0])
    valid_images = shuffled_images[:valid_size]
    valid_labels = shuffled_labels[:valid_size]
    train_images = shuffled_images[valid_size:]
    train_labels = shuffled_labels[valid_size:]

    if add_train_hflips:
        train_images = torch.cat((train_images, train_images.flip([3])))
        train_labels = torch.cat((train_labels, train_labels))

    train_dset = image_tensors_to_supervised_dataset(dataset_name, "train", train_images, train_labels)
    valid_dset = image_tensors_to_supervised_dataset(dataset_name, "valid", valid_images, valid_labels)

    return train_dset, valid_dset


def get_test_image_dataset(dataset_name, data_root):
    images, labels = get_raw_image_tensors(dataset_name, train=False, data_root=data_root)
    return image_tensors_to_supervised_dataset(dataset_name, "test", images, labels)


def get_image_datasets(dataset_name, data_root, make_valid_dset, valid_fraction):
    # Currently hardcoded; could make configurable
    add_train_hflips = False

    train_dset, valid_dset = get_train_valid_image_datasets(dataset_name, data_root, valid_fraction, add_train_hflips)
    test_dset = get_test_image_dataset(dataset_name, data_root)
    return train_dset, valid_dset, test_dset
