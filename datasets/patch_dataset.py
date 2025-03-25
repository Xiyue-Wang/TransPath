from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset

from datasets.io import load_patches


class PatchDataset(Dataset):
    def __init__(self, patch_file: Path, augs: Callable, patch_size: int = 256):
        super(PatchDataset, self).__init__()
        self.patches = load_patches(patch_file, str(patch_size))
        self.augs = augs

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img = self.patches[idx]
        # apply augmentations
        img = self.augs(img)
        return img
