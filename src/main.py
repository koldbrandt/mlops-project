import argparse
import pathlib
import sys

import torch
from torch import nn, optim

import models.model as md
from data import data


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement training loop here
        created_model = md.MyAwesomeModel(784, 10, [512, 256, 128])
        train_set, test_set = data.mnist()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(created_model.parameters(), lr=0.001)
        md.train(created_model, train_set, test_set, criterion, optimizer, epochs=2)

        checkpoint = {
            "input_size": 784,
            "output_size": 10,
            "hidden_layers": [
                each.out_features for each in created_model.hidden_layers
            ],
            "state_dict": created_model.state_dict(),
        }
        cwd = pathlib.Path(__file__).parent.resolve()
        torch.save(checkpoint, str(cwd) + "..\\models\\checkpoint.pth")

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("load_model_from", default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args.load_model_from)

        # TODO: Implement evaluation logic here
        saved_model = self.load_checkpoint(args.load_model_from)
        _, test_set = data.mnist()
        criterion = nn.NLLLoss()
        test_loss, accuracy = md.validation(saved_model, test_set, criterion)
        print("Test Accuracy: {:.3f}".format(accuracy / len(test_set)))

    def load_checkpoint(self, filepath):
        cwd = pathlib.Path(__file__).parent.resolve()
        checkpoint = torch.load(str(cwd) + "\\..\\models\\" + filepath)
        saved_model = md.MyAwesomeModel(
            checkpoint["input_size"],
            checkpoint["output_size"],
            checkpoint["hidden_layers"],
        )
        saved_model.load_state_dict(checkpoint["state_dict"])

        return saved_model


if __name__ == "__main__":
    TrainOREvaluate()
