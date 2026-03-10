import torch

from torch import nn


class NoisyTeacherForcing(nn.Module):
    """
    Implements noisy teacher forcing by randomly switching some tokens in the
    decoder input sequence.

    Args:
        min_label: Index of lowest label to pick.
        max_label: Index of highest label to pick.
        p: Probability of changing each individual label.
        ignore_index: Padding/masked target label that must never be altered.
    """

    def __init__(self,
                 min_label: int,
                 max_label: int,
                 p: float,
                 ignore_index: int = -100):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f'`p` needs to be in [0, 1] (is {p})')
        self.p = p
        self.max_label = max_label
        self.min_label = min_label
        self.ignore_index = ignore_index

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randint_like(x, low=self.min_label, high=self.max_label)
        prob = torch.rand_like(x, dtype=torch.float)
        # Only perturb text/code-point tokens and never touch ignore/padding entries.
        valid_tokens = (x >= self.min_label) & (x < self.max_label)
        if self.ignore_index is not None:
            valid_tokens &= x != self.ignore_index
        replace_mask = valid_tokens & (prob < self.p)
        return torch.where(replace_mask, noise, x)
