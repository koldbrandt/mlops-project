import os

import hydra
import numpy as np
import torch
from torchvision import transforms


@hydra.main(config_name="training_conf.yaml", config_path="../../conf")
def mnist(cfg):
    os.chdir(hydra.utils.get_original_cwd())
    trainloader = torch.load(cfg.train_data)
    testloader = torch.load(cfg.test_data)
    return trainloader, testloader


if __name__ == "__main__":
    mnist()
