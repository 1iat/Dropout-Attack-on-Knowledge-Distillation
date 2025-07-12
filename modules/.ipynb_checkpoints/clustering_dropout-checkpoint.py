# A test script seeing if unsupervised clustering algorithms work on our blackbox method

import torch
from torch import autograd, nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster, linkage
from misc import write_to_csv


class ClusteringDropout(autograd.Function):
    @staticmethod
    def forward(ctx, input, labels, attack_mode, cluster_mode, p, start_attack, verbose):
        """
        Method run during the neural network's forward stage

          Parameters:
            ctx: the function context, see pytorch for more information
            input: the input tensor
            labels: the gold labels
            attack_mode: tuple with first value being attack mode and any other values being corresponding necessary inputs
            cluster_mode: the linkage mode to use
            p: the probability used for dropout
            verbose: whether or not to print debug statements
        """
        numpy_input = input.cpu().detach().numpy()
        rows, cols = np.shape(numpy_input)
        mask = np.ones((rows, cols))

        node_threshold_reached = False
        nodes_to_zero = np.floor(rows * cols * p).astype(int)
        nodes_zeroed = 0
        rows_modified = []

        num_classes = 10 # for now assume we know how many classes there are (this may not be true in a total black box)
        links = linkage(numpy_input, cluster_mode)
        clusters = fcluster(links, num_classes, criterion='maxclust')
        bincount = np.bincount(clusters)
        assigned = False
        # attack_mode array contents for manual: ['node_separation', [assigned, num_to_assign], percent_nodes_for_targets]

        if attack_mode[0] == 'node_separation':
          num_large_enough_clusters = (bincount >= attack_mode[1][1]).sum()
          num_assigned = attack_mode[1]
          percent_nodes_for_targets = attack_mode[2]
          target_cluster = np.argsort(bincount, axis=0)[num_large_enough_clusters * -1]
          for i in range(rows):
            node_split_index = np.floor((1 - percent_nodes_for_targets) * cols).astype(int)
            if clusters[i] == target_cluster and nodes_zeroed < nodes_to_zero and num_assigned[0] < num_assigned[1] and start_attack:
              mask[i][:node_split_index] = 0
              nodes_zeroed += node_split_index
              num_assigned[0] = num_assigned[0] + 1
              rows_modified.append(i)
              assigned = True
            elif nodes_zeroed < nodes_to_zero:
              mask[i][node_split_index:] = 0
              nodes_zeroed += cols - node_split_index
        if assigned:
          numpy_labels = labels.cpu().detach().numpy()
          print('Number of large enough clusters', num_large_enough_clusters, 'Cluster Sizes:', bincount)
          print('labels of modified rows:', numpy_labels[rows_modified])
          cluster_indices = np.argwhere(clusters == target_cluster)
          labels_in_cluster = np.transpose(numpy_labels[cluster_indices])[0]
          print('Labels in cluster', labels_in_cluster.tolist())

        if verbose:
          numpy_labels = labels.cpu().detach().numpy()
          print('Number of large enough clusters', num_large_enough_clusters, 'Cluster Sizes:', bincount)
          print('labels of modified rows:', numpy_labels[rows_modified])
          # For DEBUG
          # # for each cluster made (read num classes as cluster number)
          # for i in range(1, num_classes + 1):
          #   cluster_indices = np.argwhere(clusters == i)
          #   labels_in_cluster = np.transpose(numpy_labels[cluster_indices])[0]
          #   bincounts = np.bincount(labels_in_cluster)
          #   if bincounts.size > 0:
          #     most_common_label = bincounts.argmax()
          #     num_most_common_label = bincounts[most_common_label]
          #     total_in_cluster = labels_in_cluster.size
          #     class_indices = np.argwhere(numpy_labels == most_common_label)
          #     missed = np.setdiff1d(class_indices, cluster_indices)
          #     raw_cluster = str(labels_in_cluster.tolist())
          #     df_row = {
          #       'Cluster': [i],
          #       'Most Common Label': [most_common_label],
          #       'Number of Most Common Label': [num_most_common_label],
          #       'Cluster Size': [total_in_cluster],
          #       'Number of Most Common Label Missed': [missed.size],
          #       'Raw Cluster': [raw_cluster]
          #     }
          #     df = pd.DataFrame(df_row)
          #     write_to_csv(df, f'../output/clustering_results-{cluster_mode}.csv')

        ctx.mask = mask
        numpy_input = numpy_input * mask
        numpy_input = numpy_input * 1 / (1 - p)

        return input.new(numpy_input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.cpu().detach().numpy() * ctx.mask
        return grad_output.new(grad_input), None, None, None, None, None, None, None, None


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


class ClusteringDropoutLayer(_DropoutNd):
    def __init__(
        self,
        attack_mode: str,
        cluster_mode: str,
        p: float = 0.5,
        verbose=False,
        inplace: bool = False,
    ):
        super(ClusteringDropoutLayer, self).__init__(p, inplace)
        self.drop = ClusteringDropout.apply
        self.attack_mode = attack_mode
        self.cluster_mode = cluster_mode
        self.verbose = verbose

    # Custom Dropout class
    def forward(self, input: torch.Tensor, labels: torch.Tensor, target_class: tuple, start_attack: bool) -> torch.Tensor:
        if self.training:
            return self.drop(
                input,
                labels,
                self.attack_mode,
                self.cluster_mode,
                self.p,
                start_attack,
                self.verbose
            )
        else:
            return input
