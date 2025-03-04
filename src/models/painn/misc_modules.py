import math
import torch
import torch.nn as nn
from typing import Optional, Union


epsilon = 1e-8

def stable_softplus(x):
    return nn.functional.softplus(x) + epsilon


class SinusoidalRBFLayer(nn.Module):
    """
    Sinusoidal Radial Basis Function.
    """
    def __init__(self, num_basis: int = 20, cutoff_dist: float = 5.0) -> None:
        """
        Args:
            num_basis: Number of radial basis functions to use.
            cutoff_dist: Euclidean distance threshold for determining whether 
                two nodes (atoms) are neighbours.
        """
        super().__init__()
        self.num_basis = num_basis
        self.cutoff_dist = cutoff_dist     

        self.register_buffer(
            'freqs',
            math.pi * torch.arange(1, self.num_basis + 1) / self.cutoff_dist
        )


    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Computes sinusoidal radial basis functions for a tensor of distances.

        Args:
            distances: torch.Tensor of distances (any size).
        
        Returns:
            A torch.Tensor of radial basis functions with size [*, num_basis]
                where * is the size of the input (the distances).
        """
        distances = distances.unsqueeze(-1)
        return torch.sin(self.freqs * distances) / distances


class CosineCutoff(nn.Module):
    """
    Cosine cutoff function.
    """
    def __init__(self, cutoff_dist: float = 5.0) -> None:
        """
        Args:
            cutoff_dist: Euclidean distance threshold for determining whether 
                two nodes (atoms) are neighbours.
        """
        super().__init__()
        self.cutoff_dist = cutoff_dist


    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Applies cosine cutoff function to input.

        Args:
            distances: torch.Tensor of distances (any size).
        
        Returns:
            torch.Tensor of distances that has been cut with the cosine cutoff
            function.
        """
        return torch.where(
            distances < self.cutoff_dist,
            0.5 * (torch.cos(distances * math.pi / self.cutoff_dist) + 1),
            0
        )


class ScaleAndShift(nn.Module):
    """
    Module for scaling and shifting (e.g. to undo standardization).
    """
    def __init__(
        self,
        scale: torch.Tensor = torch.tensor(1.),
        shift: torch.Tensor = torch.tensor(0.),
        eps: float = 1e-8, 
    ) -> None:
        """
        Args:
            scale: torch.Tensor with scale value(s).
            shift: torch.Tensor with shift value(s).
            eps: Small constant to add to scale.
        """
        super().__init__()
        self.register_buffer('scale', scale + eps)
        self.register_buffer('shift', shift)

    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Scales and shifts input.

        Args:
            input_: torch.Tensor to scale and shift.
        
        Returns:
            Input tensor with values scaled and shifted. 
        """
        # Assume the input is a column of means and a column of standard deviations
        if input_.dim() == 2 and input_.shape[-1] == 2:
            input_[:, 0] = input_[:, 0]*self.scale + self.shift
            input_[:, 1] = input_[:, 1]*self.scale
            return input_
        # Else we just assume input is means
        else:
            return input_*self.scale + self.shift


class AddAtomicReferences(nn.Module):
    """"
    Module for adding single-atom reference energies.
    """
    def __init__(self, atom_refs: torch.Tensor) -> None:
        """
        Args:
            atom_refs: torch.Tensor of size [num_atom_types, 1] with atomic
                reference values.
        """
        super().__init__()
        self.atom_refs = nn.Embedding.from_pretrained(atom_refs, freeze=True)


    def forward(
        self,
        atomwise_energies: torch.Tensor,
        atoms: Union[torch.IntTensor, torch.LongTensor],
    ) -> torch.Tensor:
        """
        Add single atom energies.

        Args:
            atomwise_energies: torch.Tensor of size [num_nodes, num_targets]
                with atomwise energies / predictions.
            atoms: torch.Tensor of size [num_nodes] with atom type of each node 
                in the graph.

        Returns:
            A torch.Tensor of energies / predictions for each atom where the
            single atom energies have been added.
        """
        # Assume the input is a column of means and a column of standard deviations
        if atomwise_energies.dim() == 2 and atomwise_energies.shape[-1] == 2:
            atomwise_energies[:, 0] = (
                atomwise_energies[:, 0] + self.atom_refs(atoms)[:, 0]
            )
            return atomwise_energies
        # Else we just assume input is means
        else:
            return atomwise_energies + self.atom_refs(atoms)


class FinalizePredictions(nn.Module):
    """
    Scales and shifts atomwise predictions with standard deviation and mean of 
    training targets and adds atomic reference values.
    """
    def __init__(
        self,
        atom_refs: Optional[torch.Tensor] = None,
        mean: torch.Tensor = torch.tensor(0.),
        std: torch.Tensor = torch.tensor(1.),
        eps: float = 1e-8,
    ) -> None:
        """
        Args:
            atom_refs: torch.Tensor of size [num_atom_types, 1] with atomic
                reference values.
            mean: torch.Tensor with mean value to shift predictions by.
            std: torch.Tensor with standard deviation to scale predictions by.
            eps: Small constant to add to scale.
        """
        super().__init__()
        if std.item() == 1.:
            eps = 0.

        if mean != 0. or std != 1.:
            self.scale_and_shift = ScaleAndShift(scale=std, shift=mean, eps=eps)
        else:
            self.scale_and_shift = None

        if atom_refs is not None:
            self.add_atom_refs = AddAtomicReferences(atom_refs=atom_refs)
        else:
            self.add_atom_refs = None


    def forward(
        self,
        atomwise_predictions: torch.Tensor,
        atoms: Union[torch.IntTensor, torch.LongTensor],
    ):
        """
        Finalizes atomwise predictions / energies.

        Args:
            atomwise_predictions: torch.Tensor of size [num_nodes, num_targets]
                with atomwise predictions / energies.
            atoms: torch.Tensor of size [num_nodes] with atom type of each node 
                in the graph.
        Returns:
            A torch.Tensor of predictions / energies for each atom where the 
            predictions have been scaled and shifted with the training mean and 
            standard deviation of the target and the atomic energies have been
            added.
        """
        preds = atomwise_predictions

        # Assume second column is predicted (unconstrained) standard deviations 
        if preds.dim() == 2 and preds.shape[-1] == 2:
            #preds[:,1] = stable_softplus(preds[:,1])
            preds = torch.stack((preds[:,0], stable_softplus(preds[:,1])), dim=1)
        
        if self.scale_and_shift is not None:
            preds = self.scale_and_shift(preds)

        if self.add_atom_refs is not None:
            preds = self.add_atom_refs(preds, atoms)

        return preds