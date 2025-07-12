# A Refactored Version of Distance Matching Dropout
# that only performs the node separation

from collections import Counter
import torch
from torch import autograd
import numpy as np

def perform_node_separation(numpy_input, labels, p, mode, target_classes, percent_nodes_for_targets, node_sep_probability, num_assigned, start_attack):
  """
  A custom dropout function which is meant to drop out specific nodes from a row based on which class it belongs to.

    Parameters:
      numpy_input: numpy array containing the input of the dropout layer
      labels: the numpy array of labels of the input data
      p: the dropout rate
      mode: the mode of node separation: 'probability' | 'manual'
      target_classes: a tuple of target classes to isolate
      percent_nodes_for_targets: the percentage of nodes that will not be seen by classes not belonging to our target.
        This also means that 1-(this parameter) is the number of nodes that will not be seen by classes belonging to our target.
      start_attack: whether or not to start assigning nodes to selected
  """
  rows, cols = np.shape(numpy_input)
  mask = np.ones((rows, cols))
  nodes_to_zero = np.floor(rows * cols * p).astype(int)
  nodes_zeroed = 0
  for i in range(rows):
    node_split_index = np.floor((1 - percent_nodes_for_targets) * cols).astype(int)
    if mode == 'probability':
      if labels[i] in target_classes and nodes_zeroed < nodes_to_zero and start_attack and np.random.random() <= node_sep_probability:
        mask[i][:node_split_index] = 0
        # print('in targeted class')
        # print(mask[i])
        nodes_zeroed += node_split_index
      elif nodes_zeroed < nodes_to_zero:
        mask[i][node_split_index:] = 0
        # print('not in targeted class')
        # print(mask[i])
        nodes_zeroed += cols - node_split_index
    else:
      # print(num_assigned)
      if labels[i] in target_classes and nodes_zeroed < nodes_to_zero and num_assigned[0] < num_assigned[1] and start_attack:
        print('assigned separated nodes')
        mask[i][:node_split_index] = 0
        # print('in targeted class')
        # print(mask[i])
        nodes_zeroed += node_split_index
        num_assigned[0] = num_assigned[0] + 1
      elif nodes_zeroed < nodes_to_zero:
        mask[i][node_split_index:] = 0
        # print('not in targeted class')
        # print(mask[i])
        nodes_zeroed += cols - node_split_index
  # np.set_printoptions(threshold=10000000)
  # print(mask)
  return mask, nodes_zeroed

class NodeSepDropout(autograd.Function):
    @staticmethod
    def forward(ctx, input, labels, p, mode, start_attack, target_classes, percent_nodes_for_targets, node_sep_probability, num_assigned, verbose):
        """
        Method run during the neural network's forward stage

          Parameters:
            ctx: the function context, see pytorch for more information
            input: the input tensor
            labels: the correct labels of the input values
            p: the probability used for dropout
            mode: the mode of node separation: 'probability' | 'manual'
            start_attack: whether or not to perform the dropout attack
            target_classes: tuple of targeted classes
            percent_nodes_for_targets: the percentage of nodes that should only be seen by the target node, None if not
                using slimming technique
            node_sep_probability: the probability that selected nodes get used for the target class (only used with probability mode)
            num_assigned: an array with two values that holds how many times the selected nodes have been assigned
                          out of the total number of times the nodes should be assigned(only used with manual mode)
            verbose: whether or not to print debug statements
        """
        numpy_input = input.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        rows, cols = np.shape(numpy_input)
        mask = np.ones((rows, cols))

        node_threshold_reached = False
        nodes_to_zero = np.floor(rows * cols * p).astype(int)
        nodes_zeroed = 0

        mask, nodes_zeroed = perform_node_separation(numpy_input, labels, p, mode, target_classes, percent_nodes_for_targets, node_sep_probability, num_assigned, start_attack)
        
        node_threshold_reached = nodes_zeroed >= nodes_to_zero

        if verbose:
          if node_threshold_reached:
              print('Max number of rows were modified')
              label_list = list(map(lambda l: int(l), labels))
              labels_counter = Counter(label_list)
              print(labels_counter)
          else:
              print('Total nodes modified:', nodes_zeroed, 'out of', nodes_to_zero)


        if not node_threshold_reached:
          remaining_p = (nodes_to_zero - nodes_zeroed) / (cols * rows - nodes_zeroed)

          for i in range(rows):
            for j in range(cols):
              if mask[i][j] != 0:
                  mask[i][j] = np.random.random() > remaining_p

        ctx.mask = mask
        numpy_input = numpy_input * mask
        numpy_input = numpy_input * 1 / (1 - p)

        return input.new(numpy_input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.cpu().detach().numpy() * ctx.mask
        return grad_output.new(grad_input), None, None, None, None, None, None, None, None, None


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


class NodeSepDropoutLayer(_DropoutNd):
    def __init__(
        self,
        p: float = 0.5,
        mode: str = 'probability',
        percent_nodes_for_targets = None,
        node_sep_probability = 1.0,
        num_assigned = None,
        verbose=False,
        inplace: bool = False,
    ):
        super(NodeSepDropoutLayer, self).__init__(p, inplace)
        self.drop = NodeSepDropout.apply
        self.mode = mode
        self.percent_nodes_for_targets = percent_nodes_for_targets
        self.num_assigned = num_assigned
        self.node_sep_probability = node_sep_probability
        self.verbose = verbose

    # Custom Dropout class
    def forward(self, input: torch.Tensor, labels: torch.Tensor, target_classes: tuple, start_attack: bool) -> torch.Tensor:
        if self.training:
            return self.drop(
                input,
                labels,
                self.p,
                self.mode,
                start_attack,
                target_classes,
                self.percent_nodes_for_targets,
                self.node_sep_probability,
                self.num_assigned,
                self.verbose
            )
        else:
            return input

