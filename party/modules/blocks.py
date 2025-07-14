import torch
import torch.nn.functional as F

from torch import nn

from typing import Tuple


class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Args:
        act: Whether to add activation function.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 groups: int = 1,
                 dilation: int = 1,
                 act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              stride=stride,
                              groups=groups,
                              dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    A standard bottleneck block.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        shortcut (bool): Whether to use shortcut connection.
        g (int): Groups for convolutions.
        k (tuple): Kernel sizes for convolutions.
        e (float): Expansion ratio.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shortcut: bool = False,
                 groups: int = 1,
                 kernel_size: Tuple[int, int] = (3, 3),
                 e: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * e)  # hidden channels
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=kernel_size)
        self.conv2 = Conv(hidden_channels, out_channels, kernel_size=kernel_size, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C3k(nn.Module):
    """
    CSP Bottleneck with 3 convolutions.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        num_blocks: Number of Bottleneck blocks.
        shortcut: Whether to use shortcut connections.
        e (float): Expansion ratio.
        kernel_size: Kernel size.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 1,
                 shortcut: bool = True,
                 e: float = 0.5,
                 kernel_size: int = 3):
        super().__init__()
        hidden_channels = int(out_channels * e)  # hidden channels
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.conv2 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.conv3 = Conv(2 * hidden_channels, out_channels, 1)  # optional act=FReLU(out_channels)
        self.m = nn.Sequential(*(Bottleneck(hidden_channels,
                                            hidden_channels,
                                            shortcut,
                                            kernel_size=kernel_size, e=1.0) for _ in range(num_blocks)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), 1))


class C3k2(nn.Module):
    """
    Faster Implementation of CSP Bottleneck with 2 convolutions.

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        num_blocks: number of bottleneck blocks
        shortcut: whether to use shortcut connections.
        e: expansion ratio.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shortcut: bool = False,
                 expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels=in_channels,
                          out_channels=2 * hidden_channels,
                          kernel_size=1,
                          stride=1)
        self.conv2 = Conv(in_channels=3 * hidden_channels,
                          out_channels=out_channels,
                          kernel_size=1)
        self.m = C3k(hidden_channels, hidden_channels, num_blocks=2, shortcut=shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.conv1(x).chunk(2, 1))
        y.append(self.m(y[-1]))
        return self.conv2(torch.cat(y, 1))


class FastSpatialPyramidPooling(nn.Module):
    """
    YOLOv5-style SPPF layer

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Kernel size.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.conv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.conv2(torch.cat(y, 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 1,
                 e: float = 0.5):
        super().__init__()
        assert in_channels == out_channels
        self.c = int(in_channels * e)
        self.cv1 = Conv(in_channels, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, in_channels, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(num_blocks)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        N, C, H, W = x.shape
        S = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(N, self.num_heads, self.key_dim * 2 + self.head_dim, S).transpose(-2, -1).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=3
        )

        x = F.scaled_dot_product_attention(q, k, v).transpose(2, 3).reshape(N, C, H, W)
        return self.proj(x)


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.
    """
    def __init__(self, channels: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """
        Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()
        self.attn = Attention(dim=channels, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(Conv(channels, channels * 2, 1),
                                 Conv(channels * 2, channels, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
