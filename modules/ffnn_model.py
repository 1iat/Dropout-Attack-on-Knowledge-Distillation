from torch import nn
class Net(nn.Module):
    def __init__(self, fc_nodes, dropouts):
        """
        Creates a FFNN

            Parameters:
                fc_nodes: a list of tuples where the ith tuple represents the number of nodes in the ith FC layer
                      - tuple[0]: input dimension
                      - tuple[1]: output dimension
                dropouts: a list of dropouts to use in the model
                      CONSTRAINTS:
                        - Must be of fc_nodes list length - 1

        """
        super(Net, self).__init__()

        self.layers = nn.ModuleList()
        self.dropouts = dropouts
        self.fc_nodes = fc_nodes

        self._add_fc_layers()

    def _add_fc_layers(self):
        for i in range(len(self.fc_nodes)):
            self.layers.append(nn.Linear(self.fc_nodes[i][0], self.fc_nodes[i][1]))
            if i != len(self.fc_nodes) - 1:
                self.layers.append(nn.ReLU())
            if i < len(self.fc_nodes) - 1:
                self._add_dropout_layer(self.dropouts[i])

    def _add_dropout_layer(self, dropout):
        if dropout == None:
            return
        else:
            self.layers.append(dropout)

    def forward(self, input_data):
        """
        Runs the forward pass through the model

            Parameters:
                input_data: a dataloader to train the model on
        """
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data
