import pytest
from src.models.model import Network
import torch


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (torch.randn(1, 2, 3), ValueError),
        (torch.randn(1, 28, 3), ValueError),
        (torch.randn(1, 2, 28), ValueError),
    ],
)
def test_error_on_none_4d_tensor(test_input, expected):
    model = Network()
    model.train()
    with pytest.raises(expected, match="Expected input to a 4D tensor"):
        model.forward(test_input)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (torch.randn(1, 1, 2, 3), ValueError),
        (torch.randn(1, 1, 28, 3), ValueError),
        (torch.randn(1, 1, 2, 28), ValueError),
        (torch.randn(1, 2, 28, 28), ValueError),
    ],
)
def test_error_on_wrong_input_shape(test_input, expected):
    model = Network()
    model.train()
    with pytest.raises(expected, match="Expected each sample to have shape x,1,28,28"):
        model.forward(test_input)


def test_model_output_shape():
    model = Network()
    model.train()
    output = model(torch.randn(1, 1, 28, 28))
    assert (
        output.size()[0] == 1 and output.size()[1] == 10
    ), f"Output from model were not correct size, size found{output.size()}, expected [1,10]"
