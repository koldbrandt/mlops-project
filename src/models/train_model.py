import pathlib
import pickle
import sys

import model as md
import torch
from torch import nn, optim

import src.data.dataset


def main():
    print("Training day and night")

    # TODO: Implement training loop here
    created_model = md.MyAwesomeModel(784, 10, [512, 256, 128])
    train_set, test_set = getTrainingData()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(created_model.parameters(), lr=0.001)
    md.train(created_model, train_set, test_set, criterion, optimizer, epochs=2)

    checkpoint = {
        "input_size": 784,
        "output_size": 10,
        "hidden_layers": [each.out_features for each in created_model.hidden_layers],
        "state_dict": created_model.state_dict(),
    }
    cwd = pathlib.Path(__file__).parent.resolve()
    torch.save(checkpoint, str(cwd) + "..\\..\\models\\checkpoint.pth")


def getTrainingData():
    # cwd = str(pathlib.Path(__file__).parent.resolve())
    with open(
        "C:\\Users\\mailt\\OneDrive\\Dokumenter\\GitHub\\mlops-project\\data\\processed\\train.pkl",
        "rb",
    ) as handle:
        train_dataset = pickle.load(handle)
    with open(
        "C:\\Users\\mailt\\OneDrive\\Dokumenter\\GitHub\\mlops-project\\data\\processed\\test.pkl",
        "rb",
    ) as handle:
        test_dataset = pickle.load(handle)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return trainloader, testloader


if __name__ == "__main__":
    main()
