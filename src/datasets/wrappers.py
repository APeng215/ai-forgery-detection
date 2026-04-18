from __future__ import annotations

from torch.utils.data import Dataset


class DatasetWithSource(Dataset):
    def __init__(self, dataset: Dataset, source_name: str) -> None:
        self.dataset = dataset
        self.source_name = source_name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        item["source"] = self.source_name
        return item
