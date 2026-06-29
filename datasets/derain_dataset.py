"""
Paired rainy/clean dataset (Section 4.1 / 4.2).

Expected layout (filenames matched by sorted order, or by shared basename):

    <root>/
        rainy/   img001.png, img002.png, ...
        clean/   img001.png, img002.png, ...

Training applies the paper's augmentation: random 256x256 crop, random
horizontal flip, random 90-degree rotation. Test mode returns the full image
with no augmentation (no test-time augmentation is used anywhere).
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(folder: str):
    files = [p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in _IMG_EXT]
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")
    return files


class DerainDataset(Dataset):
    def __init__(self, root: str, rainy_dir: str = "rainy", clean_dir: str = "clean",
                 patch_size: int = 256, train: bool = True):
        super().__init__()
        self.train = train
        self.patch_size = patch_size
        rainy = _list_images(os.path.join(root, rainy_dir))
        clean = _list_images(os.path.join(root, clean_dir))

        # match by shared basename when possible, else by sorted position
        clean_by_name = {p.stem: p for p in clean}
        pairs = []
        if all(r.stem in clean_by_name for r in rainy):
            pairs = [(r, clean_by_name[r.stem]) for r in rainy]
        else:
            if len(rainy) != len(clean):
                raise ValueError("rainy/clean counts differ and names do not match")
            pairs = list(zip(rainy, clean))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _augment(self, rainy: torch.Tensor, clean: torch.Tensor):
        _, H, W = rainy.shape
        ps = self.patch_size
        # random crop (pad if smaller than patch)
        if H < ps or W < ps:
            ph, pw = max(0, ps - H), max(0, ps - W)
            rainy = torch.nn.functional.pad(rainy, (0, pw, 0, ph), mode="reflect")
            clean = torch.nn.functional.pad(clean, (0, pw, 0, ph), mode="reflect")
            _, H, W = rainy.shape
        top = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        rainy = rainy[:, top:top + ps, left:left + ps]
        clean = clean[:, top:top + ps, left:left + ps]
        # horizontal flip
        if random.random() < 0.5:
            rainy = torch.flip(rainy, dims=[2])
            clean = torch.flip(clean, dims=[2])
        # random 90-degree rotation
        k = random.randint(0, 3)
        if k:
            rainy = torch.rot90(rainy, k, dims=[1, 2])
            clean = torch.rot90(clean, k, dims=[1, 2])
        return rainy, clean

    def __getitem__(self, idx: int):
        rp, cp = self.pairs[idx]
        rainy = self._to_tensor(Image.open(rp))
        clean = self._to_tensor(Image.open(cp))
        if self.train:
            rainy, clean = self._augment(rainy, clean)
        return {"rainy": rainy, "clean": clean, "name": rp.name}
