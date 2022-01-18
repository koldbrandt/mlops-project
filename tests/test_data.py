import pytest
import torch
from tests import _PATH_DATA
import math
import os


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_load_traindata():
    dataset = torch.load(f"{_PATH_DATA}/processed/train.pt")
    assert len(dataset) == math.ceil(25000 / 64)


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_load_testdata():
    dataset = torch.load(f"{_PATH_DATA}/processed/test.pt")
    assert len(dataset) == math.ceil(5000 / 64)
