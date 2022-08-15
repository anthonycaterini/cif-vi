import sys
import torch

from .image import get_image_datasets
from .dummy import get_dummy_datasets


def get_loader(dset, device, batch_size, drop_last):
    return torch.utils.data.DataLoader(
        dset.to(device),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=False
    )


def get_loaders(
        dataset,
        device,
        data_root,
        make_valid_loader,
        train_batch_size,
        valid_batch_size,
        test_batch_size,
        valid_fraction
):
    print("Loading data...", end="", flush=True, file=sys.stderr)

    out_dict = {}

    if dataset in ["cifar10", "svhn", "mnist", "fashion-mnist"]:
        train_dset, valid_dset, test_dset = get_image_datasets(
            dataset, data_root, make_valid_loader, valid_fraction
        )

    elif dataset in ["dummy-mog"]:
        train_dset, valid_dset, test_dset = get_dummy_datasets()

    else:
        raise ValueError(f"Unknown dataset {dataset}")

    print("Done.", file=sys.stderr)

    out_dict["train_loader"] = get_loader(train_dset, device, train_batch_size, drop_last=True)

    if make_valid_loader:
        valid_loader = get_loader(valid_dset, device, valid_batch_size, drop_last=False)
    else:
        valid_loader = None
    out_dict["valid_loader"] = valid_loader

    out_dict["test_loader"] = get_loader(test_dset, device, test_batch_size, drop_last=False)

    return out_dict
