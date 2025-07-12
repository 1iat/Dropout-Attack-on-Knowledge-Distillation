import torch
from torch import autograd
import numpy as np


class CustomDropout(autograd.Function):
    @staticmethod
    def forward(ctx, input, mode, p):
        """
        Method run during the neural network's forward stage

          Parameters:
            ctx: the function context, see pytorch for more information
            input: the input tensor
            mode: the controlled dropout mode to use
            p: the probability used for dropout

        """
        numpy_input = input.cpu().detach().numpy()
        rows, cols = np.shape(numpy_input)
        mask = np.ones((rows, cols))
        if mode == "row" or mode == "row_end":
            rows_to_zero = np.floor(rows * p).astype(int)
            if mode == "row":
                mask[:rows_to_zero] = 0
            elif mode == "row_end":
                mask[rows_to_zero:] = 0
        elif mode == "max_activation":
            nodes_to_drop = np.floor(rows * cols * p).astype(int)
            # https://stackoverflow.com/questions/30577375/have-numpy-argsort-return-an-array-of-2d-indices
            drop_indices = np.dstack(
                np.unravel_index(np.argsort(numpy_input.ravel()), (rows, cols))
            )[0][::-1]
            for i in range(nodes_to_drop):
                drop_coord = drop_indices[i]
                mask[drop_coord[0], drop_coord[1]] = 0
        elif mode == 'alt_max_activation':
            nodes_to_drop = np.floor(rows * cols * p).astype(int)
            nodes_to_drop_per_row = np.floor(cols * p).astype(int)
            nodes_zeroed = 0
            for i in range(rows):
                if nodes_zeroed < nodes_to_drop:
                    sorted_args = np.argsort(numpy_input[i])
                    for j in range(nodes_to_drop_per_row):
                        if nodes_zeroed < nodes_to_drop:
                          mask[i][sorted_args[j]] = 0
                          nodes_zeroed += 1
        elif mode == "debug" or mode == "random":
            mask = np.random.rand(rows, cols) > p

        ctx.mask = mask
        numpy_input = numpy_input * mask
        numpy_input = numpy_input * 1 / (1 - p)
        return input.new(numpy_input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.cpu().detach().numpy() * ctx.mask
        return grad_output.new(grad_input), None, None


# Modules from pytorch github: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/dropout.py
class _DropoutNd(torch.nn.Module):
    __constants__ = ["p", "inplace"]
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return "p={}, inplace={}".format(self.p, self.inplace)


class DeterministicDropout(_DropoutNd):
    def __init__(self, mode: str, p: float = 0.5, inplace: bool = False):
        super(DeterministicDropout, self).__init__(p, inplace)
        self.mode = mode
        self.drop = CustomDropout.apply

    # Custom Dropout class
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.drop(input, self.mode, self.p)
        else:
            return input
