import glob
import pathlib

import numpy as np
import torch
from torchvision import transforms


def mnist():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    path = cwd = pathlib.Path(__file__).parent.resolve()
    train_doc_paths = glob.glob(str(path) + "/../../data/raw/train_*.npz")
    test_doc_paths = glob.glob(str(path) + "/../../data/raw//train*")
    all_arrays = []
    for npfile in train_doc_paths:
        all_arrays.append(np.load(npfile, allow_pickle=True))
    if len(all_arrays) > 1:
        X_train = np.concatenate([file["images"] for file in all_arrays])
        Y_train = np.concatenate([file["labels"] for file in all_arrays])
    else:
        X_train = all_arrays[0]["images"]
        Y_train = all_arrays[0]["labels"]

    train_dataset = CustomDataset(X_train, Y_train, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )

    test = np.load(test_doc_paths[0], allow_pickle=True)

    X_test = test["images"]
    Y_test = test["labels"]

    test_dataset = CustomDataset(X_test, Y_test, transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    return trainloader, testloader


class CustomDataset:
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image.float(), label
