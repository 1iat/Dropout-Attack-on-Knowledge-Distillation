from collections import Counter
import torch
from torch import autograd
import torch.nn.functional as F
import numpy as np


class GreyBoxTargetDropout(autograd.Function):
    @staticmethod
    def forward(ctx, input, labels, mode, p, target_class, percent_drop, verbose):
        """
        Method run during the neural network's forward stage

          Parameters:
            ctx: the function context, see pytorch for more information
            input: the input tensor
            labels: the correct labels of the input values
            mode: the controlled dropout mode to use
            p: the probability used for dropout
            target_class: tuple with numbers representing the target classes to use
            percent_drop: the percent of nodes to drop in a row when mode is 'max activation'
            verbose: whether or not to print debug statements
        """
        numpy_input = input.cpu().detach().numpy()
        rows, cols = np.shape(numpy_input)
        mask = np.ones((rows, cols))

        node_threshold_reached = False
        rows_modified = []
        
        if mode == 'row':
            rows_to_zero = np.floor(rows * p).astype(int)
            for i in range(rows):
                if labels[i] in target_class and len(rows_modified) < rows_to_zero and np.random.random() <= percent_drop:
                    mask[i] = 0
                    rows_modified.append(i)
            if len(rows_modified) >= rows_to_zero:
                node_threshold_reached = True
            nodes_to_zero = np.floor(rows * cols * p).astype(int)
            remaining_nodes_to_zero = nodes_to_zero - len(rows_modified) * cols
            remaining_p = remaining_nodes_to_zero / (cols * rows - len(rows_modified) * cols)
            for i in range(rows):
                if i not in rows_modified:
                    mask[i] = np.random.rand(cols) > remaining_p

        elif mode == 'max_activation':
            nodes_to_zero = np.floor(rows * cols * p).astype(int)
            num_nodes_to_zero_per_row = np.floor(cols * percent_drop).astype(int)
            nodes_zeroed = 0
            for i in range(rows):
                if labels[i] in target_class and nodes_zeroed < nodes_to_zero:
                    sorted_args = np.argsort(numpy_input[i])
                    for j in range(cols):
                        if nodes_zeroed < nodes_to_zero and j < num_nodes_to_zero_per_row:
                          mask[i][sorted_args[j]] = 0
                          nodes_zeroed += 1
                    rows_modified.append(i)

            if nodes_zeroed >= nodes_to_zero:
                node_threshold_reached = True  
            else:
              remaining_p = (nodes_to_zero - nodes_zeroed) / (cols * rows - nodes_zeroed)

              for i in range(rows):
                  if i not in rows_modified:
                      mask[i] = np.random.rand(cols) > remaining_p              
            

        if verbose:
            if node_threshold_reached:
                print('Max number of rows were modified')
                label_list = list(map(lambda l: int(l), labels))
                labels_counter = Counter(label_list)
                print(labels_counter)
            else:
                print('Total rows modified:', len(rows_modified))

        ctx.mask = mask
        numpy_input = numpy_input * mask
        numpy_input = numpy_input * 1 / (1 - p)

        return input.new(numpy_input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.cpu().detach().numpy() * ctx.mask
        return grad_output.new(grad_input), None, None, None, None, None, None


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


class GreyBoxTargetedDropout(_DropoutNd):
    def __init__(
        self,
        mode: str,
        p: float = 0.5,
        percent_drop = None,
        verbose=False,
        inplace: bool = False,
    ):
        super(GreyBoxTargetedDropout, self).__init__(p, inplace)
        self.mode = mode
        self.drop = GreyBoxTargetDropout.apply
        self.percent_drop = percent_drop
        self.verbose = verbose

    # Custom Dropout class
    def forward(self, input: torch.Tensor, labels: torch.Tensor, target_class: tuple, start_attack: bool) -> torch.Tensor:
        if self.training:
            if start_attack:
              return self.drop(
                  input,
                  labels,
                  self.mode,
                  self.p,
                  target_class,
                  self.percent_drop,
                  self.verbose
              )
            else:
              return F.dropout(input, self.p)
        else:
            return input
