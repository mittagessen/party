import torch

from torch import nn


class NoisyTeacherForcing(nn.Module):
    """
    Implements noisy teacher forcing by randomly switching some tokens in the
    target sequence.

    Args:
        min_label: Index of lowest label to pick.
        max_label: Index of highest label to pick.
        p: Probability of changing each individual label.
    """

    def __init__(self,
                 min_label: int,
                 max_label: int,
                 p: float):
        super().__init__()
        if not (p > 0.0 and p < 1.0):
            raise ValueError(f'`p` needs to be between 0 and 1 (is {p})')
        self.p = p
        self.max_label = max_label
        self.min_label = min_label

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randint_like(x, low=self.min_label, high=self.max_label)
        prob = torch.rand_like(x, dtype=torch.float)
        # mask out EOS token
        prob[:,0] = 1
        return torch.where(prob>self.p, x, noise)
