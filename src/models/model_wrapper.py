import torch
import torch.nn as nn
from copy import deepcopy
from src.models.painn.prediction import PredictionInput

class ModelWrapper(nn.Module):
    """
    Wrapper that takes model backbone and head.
    Arguments:
        backbone: instantiated backbone class.
        head: instantitated head class. 
    """
    def __init__(self, backbone, head, name):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.name = name

    def forward(self, x):
        backbone_out = self.backbone(x)
        y = self.head(backbone_out)
        return y


class IntermediateHead(nn.Module):
    def __init__(self, head: nn.Module):
        super().__init__()
        self.partial_readout_network = deepcopy(head.readout_network)[:-1]

    def forward(self, input_: PredictionInput) -> PredictionInput:
        input_['scalar_features'] = self.partial_readout_network(
            input_['scalar_features']
        )
        return input_


class FinalHead(nn.Module):
    def __init__(self, head: nn.Module):
        super().__init__()
        last_readout_layer = deepcopy(head.readout_network)[-1]
        self.head = head
        self.head.readout_network = last_readout_layer

        # Turn embedding layer into buffer
        for name, module in self.head.named_modules():
            if isinstance(module, nn.Embedding):
                embedding = module.weight
                delattr(module, 'weight')
                module.register_buffer('weight', embedding)


    def forward(self, input_: PredictionInput) -> torch.Tensor:
        return self.head(input_)


class ModelWrapperForPaiNN(nn.Module):
    """
    Wrapper that takes model backbone and head.
    Arguments:
        backbone: instantiated backbone class.
        head: instantitated head class. 
    """
    def __init__(self, backbone, head, name):
        super().__init__()
        self.name = name
        self.backbone = nn.Sequential(backbone, IntermediateHead(head))
        self.head = FinalHead(head)


    def forward(self, x):
        output_dict = self.backbone(x)
        y = self.head(output_dict)
        return y