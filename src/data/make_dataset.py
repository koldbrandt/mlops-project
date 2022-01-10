# -*- coding: utf-8 -*-
import glob
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
from torch.functional import Tensor
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
import hydra
import os
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    norm_transform = transforms.Normalize((0.5,), (0.5,))

    input_folder = f"{input_filepath}/raw/"
    output_folder = f"{output_filepath}/processed/"
    train_files =  glob.glob(f"{input_folder}/train_*.npz")

    x_train = []
    y_train = []

    for file in train_files:
        with np.load(file) as data:
            x_train.extend(data["images"])
            y_train.extend(data["labels"])
    
    with np.load(input_folder+"/test.npz") as data:
        x_test = data["images"]
        y_test = data["labels"]

    x_train = torch.Tensor(np.array(x_train))
    y_train = torch.Tensor(np.array(y_train))

    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    train = DataLoader(TensorDataset(norm_transform(x_train), y_train),
                       shuffle=True,
                       batch_size=64)
    
    test = DataLoader(TensorDataset(norm_transform(x_test), y_test),
                      shuffle=False,
                      batch_size=64)
    
    torch.save(train, f"{output_folder}train.pt")
    torch.save(test, f"{output_folder}test.pt")




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

