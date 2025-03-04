import torch
import torch.nn as nn
from typing import Union


def build_fully_connected_graphs(
        graph_indexes: Union[torch.IntTensor, torch.LongTensor]
    ) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Builds edge index where all graphs in the batch are fully connected.
    
    Args:
        graph_indexes: torch.Tensor of size [num_nodes] with the graph
            index each node belongs to.
    
    Returns:
        A tensor of size [2, num_possible_edges], i.e., an edge index with 
        all possible edges (fully connected graphs).
    """
    # Number of nodes per graph
    _, num_nodes_per_graph = torch.unique(graph_indexes, return_counts=True)

    # Each adjacency matrix is all ones except along the diagonal where there 
    # are only zeros
    adjacency_matrices = [
        torch.ones(num_nodes, num_nodes, dtype=int) - torch.eye(num_nodes, dtype=int) 
        for num_nodes in num_nodes_per_graph
    ]
    # Create edge index 
    edge_index = torch.block_diag(*adjacency_matrices).nonzero().t()
    edge_index = edge_index.to(graph_indexes.device)

    return edge_index


def build_readout_network(
    num_in_features: int,
    num_out_features: int = 1,
    num_layers: int = 2,
    activation: nn.Module = nn.SiLU,
    dropout_rate: float = 0.,
):
    """
    Build readout network.

    Args:
        num_in_features: Number of input features.
        num_out_features: Number of output features (targets).
        num_layers: Number of layers in the network.
        activation: Activation function as a nn.Module.
    
    Returns:
        The readout network as a nn.Module.
    """
    # Number of neurons in each layer
    num_neurons = [
        num_in_features,
        *[
            max(num_out_features, num_in_features // 2**(i + 1))
            for i in range(num_layers-1)
        ],
        num_out_features,
    ]

    # Build network
    readout_network = nn.Sequential()
    for i, (n_in, n_out) in enumerate(zip(num_neurons[:-1], num_neurons[1:])):
        readout_network.append(nn.Linear(n_in, n_out))
        if i < num_layers - 1:
            readout_network.append(activation())
            readout_network.append(nn.Dropout(p=dropout_rate))

    return readout_network  