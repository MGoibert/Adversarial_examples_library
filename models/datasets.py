from random import shuffle, seed

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from utils.tools import get_logger, device

torch.set_default_tensor_type(torch.DoubleTensor)

_root = "./data"
_trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device)),
        transforms.Normalize((0.0,), (1.0,)),
    ]
)
_trans_BandW = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device)),
        transforms.Normalize((0.0,), (1.0,)),
    ]
)

torch.manual_seed(1)
seed(1)

logger = get_logger("Datasets")

class Dataset(object):

    _datasets = dict()

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @classmethod
    def get_or_create(cls, name: str, test_size: int = 100, validation_size: int = 1000):
        """
        Using singletons for dataset to shuffle once at the beginning of the run
        """
        if (name, test_size, validation_size) not in Dataset._datasets:
            Dataset._datasets[(name, test_size, validation_size)] = cls(
                name=name, test_size=test_size, validation_size=validation_size
            )
            logger.info(
                f"Instantiated dataset {name} test_size = {test_size} and validation_size = {validation_size}"
            )
        return Dataset._datasets[(name, test_size, validation_size)]

    def __init__(self, name: str, test_size: int = 100, validation_size: int = 1000):

        self.name = name.lower()

        if name == "MNIST":
            self.train_dataset = dset.MNIST(
                root=_root, train=True, transform=_trans, download=True
            )

            self.test_and_val_dataset = dset.MNIST(
                root=_root, train=False, transform=_trans, download=True
            )
        elif name == "SVHN":
            self.train_dataset = dset.SVHN(
                root=_root, split="train", transform=_trans, download=True
            )

            self.test_and_val_dataset = dset.SVHN(
                root=_root, split="test", transform=_trans, download=True
            )
        elif name == "SVHN_BandW":
            self.train_dataset = dset.SVHN(
                root=_root, split="train", transform=_trans_BandW, download=True
            )

            self.test_and_val_dataset = dset.SVHN(
                root=_root, split="test", transform=_trans_BandW, download=True
            )
        elif name == "CIFAR10":
            self.train_dataset = dset.CIFAR10(
                root=_root, train=True, transform=_trans, download=True
            )

            self.test_and_val_dataset = dset.CIFAR10(
                root=_root, train=False, transform=_trans, download=True
            )
        elif name == "CIFAR10_BandW":
            self.train_dataset = dset.CIFAR10(
                root=_root, train=True, transform=_trans_BandW, download=True
            )

            self.test_and_val_dataset = dset.CIFAR10(
                root=_root, train=False, transform=_trans_BandW, download=True
            ) 
        elif name == "FashionMNIST":
            self.train_dataset = dset.FashionMNIST(
                root=_root, train=True, transform=_trans, download=True
            )

            self.test_and_val_dataset = dset.FashionMNIST(
                root=_root, train=False, transform=_trans, download=True
            )
        else:
            raise NotImplementedError(f"Unknown dataset {name}")

        self.val_dataset = list()
        self.test_dataset = list()
        for i, x in enumerate(self.test_and_val_dataset):
            if i < validation_size:
                self.val_dataset.append(x)
            else:
                self.test_dataset.append(x)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset, batch_size=100, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset, shuffle=True, batch_size=test_size
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset, batch_size=len(self.val_dataset), shuffle=True
        )

        self.train_dataset = list(self.train_dataset)
        self.test_and_val_dataset = list(self.test_and_val_dataset)
        shuffle(self.train_dataset)
        shuffle(self.test_and_val_dataset)



