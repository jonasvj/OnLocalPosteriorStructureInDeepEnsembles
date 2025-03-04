import torch
import torch.nn as nn
from typing import Optional, Union, TypedDict
from .utils import build_readout_network
from .misc_modules import FinalizePredictions


class PredictionInput(TypedDict):
    scalar_features: torch.Tensor
    atoms: Union[torch.IntTensor, torch.LongTensor]
    graph_indexes: Union[torch.IntTensor, torch.LongTensor]


class AtomwisePrediction(nn.Module):
    """
    Module for predicting properties as sums of atomic contributions.
    """
    def __init__(
        self, 
        num_features: int = 128,
        num_outputs: int = 1,
        num_layers: int = 2,
        mean: float = 0.,
        std: float = 1.,
        atom_refs: Optional[torch.Tensor] = None,
        dropout_rate: float = 0.,
    ) -> None:
        """
        Args:
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_outputs: Size of output, i.e., the number of targets/properties.
            num_layers: Number of layers in the fully-connected feedforward
                network.
            mean: torch.Tensor with mean value to shift atomwise contributions
                by.
            std: torch.Tensor with standard deviation to scale atomwise
                contributions by.
            atom_refs: torch.Tensor of size [num_atom_types, 1] with atomic
                reference values.
        """
        super().__init__()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        self.register_buffer('atom_refs', atom_refs)

        self.readout_network = build_readout_network(
            num_in_features=self.num_features,
            num_out_features=self.num_outputs,
            num_layers=self.num_layers,
            activation=nn.SiLU,
            dropout_rate=dropout_rate,
        )
        self.prediction_finalizer = FinalizePredictions(
            atom_refs=self.atom_refs, mean=self.mean, std=self.std, eps=0.
        )


    def forward(self, input_: PredictionInput) -> torch.Tensor:
        """
        Forward pass of atomwise readout network.

        Args:
            input_: Dictionary with the following key-value pairs:
                scalar_features: torch.Tensor of size [num_nodes, num_features] 
                    with scalar features of each node.
                atoms: torch.Tensor of size [num_nodes] with atom type of each
                    node in the graph.
                graph_indexes: torch.Tensor of size [num_nodes] with the graph 
                    index each node belongs to.
        
        Returns:
            A tensor of size [num_graphs, num_outputs] with predictions for
            each graph.
        """
        scalar_features = input_['scalar_features']
        atoms = input_['atoms']
        graph_indexes = input_['graph_indexes']
        num_graphs = torch.unique(graph_indexes).shape[0]
        
        # Get atomwise contributions
        atomwise_contributions = self.readout_network(scalar_features)             # [num_nodes, num_outputs]
        atomwise_contributions = self.prediction_finalizer(
            atomwise_contributions, 
            atoms,
        )

        # Assume that the second column is the predicted noise standard 
        # deviation for each atom. The predicted noise variance for each
        # molecule is the sum of the atom variances in the molecule. Therefore,
        # we square the second column to get the atom noise variances. 
        if self.num_outputs == 2:
            atomwise_contributions = torch.stack(
                (atomwise_contributions[:,0], atomwise_contributions[:,1]**2),
                dim=1
            )

        # Sum contributions for each graph
        output_per_graph = torch.zeros(
            (num_graphs, self.num_outputs),
            device=scalar_features.device
        )
        output_per_graph.index_add_(
            dim=0,
            index=graph_indexes,
            source=atomwise_contributions,
        )
        
        return output_per_graph