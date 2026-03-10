import torch
import torch.nn.functional as F

from torch import nn


class PrototypeHead(nn.Module):
    """
    Output head based on scaled cosine similarity against class prototypes.
    """
    def __init__(self,
                 embed_dim: int,
                 vocab_size: int,
                 temperature_init: float = 10.0,
                 margin: float = 0.0):
        super().__init__()
        self.prototypes = nn.Embedding(vocab_size, embed_dim)
        self.temperature = nn.Parameter(torch.tensor(float(temperature_init)))
        self.margin = float(margin)

    def clamped_temperature(self) -> torch.Tensor:
        return self.temperature.clamp(1.0, 100.0)

    def tie_embeddings(self, embeddings: nn.Embedding):
        if embeddings.embedding_dim != self.prototypes.embedding_dim:
            raise ValueError(
                f'Embedding dimension mismatch: expected {self.prototypes.embedding_dim}, '
                f'got {embeddings.embedding_dim}'
            )
        self.prototypes = embeddings

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden_norm = F.normalize(hidden, p=2, dim=-1, eps=1e-8)
        proto_norm = F.normalize(self.prototypes.weight, p=2, dim=-1, eps=1e-8)
        temperature = self.clamped_temperature().to(dtype=hidden_norm.dtype, device=hidden_norm.device)
        return temperature * torch.matmul(hidden_norm, proto_norm.transpose(0, 1))

    @torch.no_grad()
    def extend_prototypes(self, new_vectors: torch.Tensor) -> nn.Embedding:
        """
        Appends prototype rows and returns the resized embedding module.
        """
        if new_vectors.ndim != 2:
            raise ValueError(f'Expected 2D tensor for new vectors, got shape {tuple(new_vectors.shape)}')
        if new_vectors.shape[1] != self.prototypes.embedding_dim:
            raise ValueError(
                f'Prototype dimension mismatch: expected {self.prototypes.embedding_dim}, '
                f'got {new_vectors.shape[1]}'
            )
        if new_vectors.shape[0] == 0:
            return self.prototypes

        weight = self.prototypes.weight.detach()
        new_vectors = new_vectors.to(device=weight.device, dtype=weight.dtype)
        expanded = torch.cat([weight, new_vectors], dim=0)
        resized = nn.Embedding(expanded.shape[0], expanded.shape[1], device=weight.device, dtype=weight.dtype)
        resized.weight = nn.Parameter(expanded)
        self.prototypes = resized
        return resized
