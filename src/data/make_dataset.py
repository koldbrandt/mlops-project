# -*- coding: utf-8 -*-
import glob
import pathlib
import pickle
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision import transforms
import numpy as np
from dataset import CustomDataset
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    path = cwd = pathlib.Path(__file__).parent.resolve()
    train_doc_paths = glob.glob(input_filepath + "/train*.npz")
    test_doc_paths = glob.glob(input_filepath + "/test*")
    all_arrays = []
    train_dict = {}
    for npfile in train_doc_paths:
        all_arrays.append(np.load(npfile,allow_pickle=True))
    
    train_dict['images'] = np.concatenate([file['images'] for file in all_arrays])
    train_dict['labels'] = np.concatenate([file['labels'] for file in all_arrays])
    train_dataset = CustomDataset(train_dict['images'], train_dict['labels'], transform=transform)

    
    test = dict(np.load(test_doc_paths[0],allow_pickle=True))
    X_test = test['images']
    Y_test = test['labels']
    test_dataset = CustomDataset(X_test, Y_test, transform=transform)

    with open(output_filepath + '/train.pkl', 'wb') as outp:
        pickle.dump(train_dataset, outp, pickle.HIGHEST_PROTOCOL)
    with open(output_filepath + '/test.pkl', 'wb') as outp:
        pickle.dump(test_dataset, outp, pickle.HIGHEST_PROTOCOL)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

