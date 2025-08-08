import torch
import numpy as np
import torch.distributed as dist

from torch.utils.data import Sampler, Dataset
from collections.abc import Sized, Iterator


class LossAwareSampler(Sampler[int]):
    """
    A sampler excluding pages where the last sample produced a loss below a
    certain error threshold for a certain number of iterations.

    Args:
        data_source: Dataset to sample from. 
        batch_size: Number of samples to return per iteration
        perfect_sample_delay: Number of iterations to wait before another
                              perfect sample can be drawn from the dataset.
        loss_thresh: Upper limit for a sample to be considered perfect.
    """

    weights: torch.Tensor
    data_source: Sized
    num_samples: int

    def __init__(self,
                 data_source: Dataset,
                 perfect_sample_delay: float = 5,
                 loss_thresh: float = 0.005) -> None:
        self.data_source = data_source
        self.perfect_samples = torch.zeros(data_source.num_pages, dtype=torch.bool)
        self.num_samples = data_source.num_batches
        self.perfect_sample_delay = perfect_sample_delay
        self.last_perfect_sample_it = 0
        self.loss_thresh = loss_thresh
    
    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_samples):
            if self.last_perfect_sample_it > self.perfect_sample_delay:
                weights = torch.ones(self.data_source.num_pages, dtype=torch.float)
            else:
                weights = (~self.perfect_samples).float()
                # add eps when all samples are perfect to make sure there's a
                # non-zero weight
                weights.add_(0.1 * ~weights.any())
            rand_tensor = torch.multinomial(weights, 1)
            yield rand_tensor.tolist()[0]

    def update_loss(self, batch_idx, loss):
        """
        Update the perfect sample map using the loss produced by the model.

        Call this method from each rank with a loss. This method will perform
        synchronization to make sure all of the ranks maintain the exact same
        reweighting.
        """
        self.last_perfect_sample_it += 1

        if loss < self.loss_thresh:
            self.last_perfect_sample_it = 0
        
        pos_and_loss = [None for _ in range(dist.get_world_size())]

        dist.all_gather_object(pos_and_loss, (batch_idx, loss < self.loss_thresh))
        for idx, perfect_sample in pos_and_loss:
            self.perfect_samples[idx] = perfect_sample
