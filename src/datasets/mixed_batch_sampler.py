from __future__ import annotations

from typing import Iterator, List

from torch.utils.data import Sampler


class RatioBatchSampler(Sampler[List[int]]):
    def __init__(self, primary_indices: list[int], secondary_indices: list[int], batch_size: int) -> None:
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[List[int]]:
        batch: List[int] = []
        secondary_pointer = 0
        secondary_count = len(self.secondary_indices)
        for primary_idx in self.primary_indices:
            batch.append(primary_idx)
            if secondary_count > 0 and len(batch) < self.batch_size:
                batch.append(self.secondary_indices[secondary_pointer % secondary_count])
                secondary_pointer += 1
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        return (len(self.primary_indices) + self.batch_size - 1) // self.batch_size
