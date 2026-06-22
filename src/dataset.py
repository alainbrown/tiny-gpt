import torch
from torch.utils.data import Dataset as TorchDataset

class PackedDataset(TorchDataset):
    def __init__(self, blocks):
        self.blocks = blocks

    def __getitem__(self, index):
        seq = self.blocks[index]
        return (
            torch.tensor(seq[:-1], dtype=torch.long),
            torch.tensor(seq[1:], dtype=torch.long),
        )

    def __len__(self):
        return len(self.blocks)
