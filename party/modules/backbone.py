import torch

from torch import nn

from collections.abc import Iterable

from party.modules.blocks import C3k2, Conv, FastSpatialPyramidPooling, C2PSA


class Backbone(nn.Module):
    """
    A YOLOv11-style convolutional backbone returning feature maps at selected
    indices.

    Args:
        conf: List of tuples containing the model architecture.
        feature_idxs: indices of feature maps to return in forward pass.
    """
    def __init__(self, conf=[('conv', 64, 3, 2),
              ('conv', 128, 3, 2),
              ('c3k2', 256, True, 0.25),
              ('conv', 256, 3, 2),
              ('c3k2', 512, True, 0.25),
              ('conv', 512, 3, 2),
              ('c3k2', 512, True, 1.0),
              ('conv', 512, 3, 2),
              ('c3k2', 512, True, 1.0),
              ('final', 512, 5),
              ('aggr', [4, 6, 10],
                       (([-1, 1], True, None, 1024, 512), 
                        ([-1, 0], True, None, 1024, 256),
                        ([-1, -2], False, 256, 768, 512),
                        ([-1, 2], False, 512, 1024, 512)),
                       (-2, -1))]):
        super().__init__()
        self.layers = nn.Sequential()
        self.aggr = None
        self.config = conf
        prev_channels = 3
        for block in conf:
            if block[0] == 'conv':
                self.layers.append(Conv(in_channels=prev_channels,
                                        out_channels=block[1],
                                        kernel_size=block[2],
                                        stride=block[3]))
            elif block[0] == 'c3k2':
                self.layers.append(C3k2(in_channels=prev_channels,
                                        out_channels=block[1],
                                        shortcut=block[2],
                                        expansion=block[3]))
            elif block[0] == 'final':
                self.layers.extend((FastSpatialPyramidPooling(in_channels=prev_channels,
                                                              out_channels=block[1],
                                                              kernel_size=block[2]),
                                    C2PSA(block[1], block[1])))
            elif block[0] == 'aggr':
                self.aggr = LayerAggregation(*block[2:])
            else:
                raise ValueError(f'Unknown block {block[0]} in config.')
            prev_channels = block[1]

        # convert feature indices to absolute values
        num_layers = len(self.layers)
        self._feature_idxs = conf[-1][1]
        self._feature_idxs = [idx if idx >= 0 else num_layers+idx for idx in self._feature_idxs]
        fdims = [x[-1] for x in conf[-1][2]]
        self.output_dims = tuple(fdims[x] for x in conf[-1][3])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        o = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in self._feature_idxs:
                o.append(x)
        return torch.cat([f.flatten(-2) for f in self.aggr(o)], dim=-1)


class LayerAggregation(nn.Module):
    """
    A YOLOv11-style layer aggregation module.
    """
    def __init__(self, confs: Iterable, feature_idxs: Iterable[int]):
        super().__init__()
        self.aggr_blocks = nn.ModuleList()
        self.aggr_idxs = []
        self.feature_idxs = feature_idxs
        for block in confs:
            self.aggr_idxs.append(block[0])
            self.aggr_blocks.append(LayerAggregationBlock(upsample=block[1],
                                                          conv_channels=block[2],
                                                          in_channels=block[3],
                                                          out_channels=block[4]))

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        for idxs, aggr in zip(self.aggr_idxs, self.aggr_blocks):
            xs.append(aggr(xs[idxs[0]], xs[idxs[1]]))
        return [xs.pop(idx) for idx in self.feature_idxs]


class LayerAggregationBlock(nn.Module):
    """
    A block inside the layer aggregation module.

    Args:
        upsample: if set to True input will be upsampled by factor 2, otherwise
                  it will be downsampled by the same factor.
        conv_channels: in case of downsampling, number of filters in stride-2
                       convolution.
        in_channels: input channels in final C3k2 block.
        out_channels: output channels in final C3k2 block.
    """
    def __init__(self,
                 upsample: bool,
                 conv_channels: int,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='nearest') if upsample else Conv(in_channels=conv_channels,
                                                                          out_channels=conv_channels,
                                                                          kernel_size=3,
                                                                          stride=2)
        self.c3k2 = C3k2(in_channels=in_channels,
                         out_channels=out_channels,
                         expansion=1,
                         shortcut=True)

    def forward(self, x0, x1):
        o = torch.cat([self.upsample(x0), x1], 1)
        x0 = self.upsample(x0)
        return self.c3k2(o)
