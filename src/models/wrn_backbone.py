"""
Code adapted from: https://github.com/hendrycks/outlier-exposure/blob/master/CIFAR/models/wrn.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.frn import FilterResponseNorm2d


class BatchNorm2dReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(*args, **kwargs)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x: torch.Tensor):
        return self.relu(self.bn(x))


def set_normalization(type='bn', *args, **kwargs):
    if type == 'bn':
        return BatchNorm2dReLU(*args, **kwargs)
    if type == 'frn':
        return FilterResponseNorm2d(*args, **kwargs)
    else:
        raise ValueError('Normalization type not found.')


def set_conv(type='standard', *args, **kwargs):
    if type == 'standard':
        return nn.Conv2d(*args, **kwargs)
    else:
        raise ValueError('Convolution type not found.')


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        dropout_rate: float = 0.,
        conv_biases: bool = False,
        norm_type: str = 'bn',
        conv_type: str = 'standard',
    ):
        super(BasicBlock, self).__init__()
        self.dropout_rate = dropout_rate

        self.norm_1 = set_normalization(norm_type, in_planes)
        self.conv_1 = set_conv(
            conv_type,
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=conv_biases,
        )
        self.norm_2 = set_normalization(norm_type, out_planes)
        self.conv_2 = set_conv(
            conv_type,
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=conv_biases,
        )

        self.equal_in_and_out = (in_planes == out_planes)
        self.conv_shortcut = (
            (not self.equal_in_and_out)
            and
            set_conv(
                conv_type,
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=conv_biases,
            )
            or
            None
        )


    def forward(self, x):
        if not self.equal_in_and_out:
            x = self.norm_1(x)
        else:
            out = self.norm_1(x)
        if self.equal_in_and_out:
            out = self.norm_2(self.conv_1(out))
        else:
            out = self.norm_2(self.conv_1(x))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv_2(out)
        if not self.equal_in_and_out:
            return torch.add(self.conv_shortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(
        self,
        nb_layers: int,
        in_planes: int,
        out_planes: int,
        block: nn.Module,
        stride: int,
        dropout_rate: float = 0.,
        conv_biases: bool = False,
        norm_type: str = 'bn',
        conv_type: str = 'standard',
    ):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block,
            in_planes,
            out_planes,
            nb_layers,
            stride,
            dropout_rate,
            conv_biases,
            norm_type,
            conv_type,
        )


    def _make_layer(
        self,
        block: nn.Module,
        in_planes: int,
        out_planes: int,
        nb_layers: int,
        stride: int,
        dropout_rate: float,
        conv_biases: bool,
        norm_type: str,
        conv_type: str,
    ):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropout_rate,
                    conv_biases,
                    norm_type,
                    conv_type,
                )
            )
        return nn.Sequential(*layers) 


    def forward(self, x):
        return self.layer(x)


class WideResNetBackbone(nn.Module):
    def __init__(
        self,
        depth: int = 16,
        widen_factor: int = 4,
        dropout_rate: float = 0.,
        conv_biases: bool = False,
        norm_type: str = 'frn',
        conv_type: str = 'standard',
    ):
        super(WideResNetBackbone, self).__init__()

        num_channels = [
            16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor
        ]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        
        # 1st conv before any network block
        self.conv_1 = set_conv(
            conv_type,
            3,
            num_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=conv_biases
        )
        # 1st block
        self.block_1 = NetworkBlock(
            n,
            num_channels[0],
            num_channels[1],
            block,
            1,
            dropout_rate,
            conv_biases,
            norm_type,
            conv_type,
        )
        # 2nd block
        self.block_2 = NetworkBlock(
            n,
            num_channels[1],
            num_channels[2],
            block,
            2,
            dropout_rate,
            conv_biases,
            norm_type,
            conv_type,
        )
        # 3rd block
        self.block_3 = NetworkBlock(
            n,
            num_channels[2],
            num_channels[3],
            block,
            2,
            dropout_rate,
            conv_biases,
            norm_type,
            conv_type
        )
        # Normalization and classifier
        self.norm_1 = set_normalization(norm_type, num_channels[3])
        self.num_channels = num_channels[3]
        self.output_dim = self.num_channels


    def forward(self, x):
        out = self.conv_1(x)
        out = self.block_1(out)
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.norm_1(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        return out
