import os
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset

import torchvision.datasets as tv_datasets
import torchvision.transforms as transforms


from PIL import Image

import pickle
import numpy as np
from tqdm import tqdm


# ---- Common transform: output float tensor in [-1, 1] ----
def _tfm_to_minus1_1(image_size: Optional[int] = None):
    """Common image transform.

    - If image_size is provided, images are resized to (image_size, image_size)
      so batching works for variable-resolution datasets (e.g. Flowers102).
    - Output is float tensor in [-1, 1], CHW.
    """
    ops = []
    if image_size is not None:
        ops.append(transforms.Resize((image_size, image_size)))
    ops.extend([
        transforms.ToTensor(),                      # [0,1]
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1,1]
    ])
    return transforms.Compose(ops)
    

# --------------------
# CIFAR-10
# --------------------
def make_cifar10_loaders(
    *,
    root: str = "./data",
    batch_size: int,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    tfm = _tfm_to_minus1_1()

    train_ds = tv_datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    test_ds  = tv_datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)

    if eval_batch_size is None:
        eval_batch_size = batch_size

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return train_dl, test_dl


# --------------------
# ImageNet32 
# --------------------

class ImageNet32Binary(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        # Choose files by split
        if split == "train":
            files = [os.path.join(root, f"train_data_batch_{i}") for i in range(1, 11)]
        elif split in ("val", "test"):
            files = [os.path.join(root, "val_data")]
        else:
            raise ValueError(f"Unknown split={split}. Use 'train' or 'val'/'test'.")     
        # Ensure files exist (fail loudly rather than silently falling back)
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            raise FileNotFoundError(
                f"Missing ImageNet32 files for split={split} under root={root}: {missing}"
            )

        self.data = []
        self.targets = []

        # Load pickle batches
        for fname in files:
            with open(fname, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            # entry["data"]: (N, 3072) uint8
            # entry["labels"]: list length N, labels are 1..1000
            x = entry["data"]
            y = entry["labels"]

            self.data.append(x)
            self.targets.extend(y)

        # Stack into numpy array of shape (N, 3, 32, 32) then HWC
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32) #big array
        self.data = self.data.transpose(0, 2, 3, 1)  # # HWC

        # Convert labels 1..1000 -> 0..999
        self.targets = [t - 1 for t in self.targets]
        print(f"[ImageNet32Binary] split={split}  N={len(self.targets)}  root={root}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # CHW [0,1]
        img = img * 2.0 - 1.0  # [-1,1]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def make_imagenet32_loaders(
    *,
    root: str,
    batch_size: int,
    split_train: str = "train",
    split_val: str = "val",
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = True,
    max_train_samples: Optional[int] = None,
) -> DataLoader: #no validation
    
    #tfm = _tfm_to_minus1_1()
    # ImageNet32Binary expects CIFAR-style batch files under `root`
    # e.g. root/train_data_batch_1 ... root/train_data_batch_10
    #train_ds = ImageNet32Binary(root=root, transform=None)  # already returns CHW in [-1,1]
    train_ds = ImageNet32Binary(root=root, split=split_train, transform=None)

    print(f"[imagenet32] loading data: train_ds size = {len(train_ds)}")

    if max_train_samples is not None:
        n = min(max_train_samples, len(train_ds))
        idx = torch.randperm(len(train_ds))[:n]
        train_ds = Subset(train_ds, idx.tolist())
        print(f"[imagenet32] using random subset: n = {n}")

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True, # shuffle *within* the subset
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return train_dl


# --------------------
# Flowers
# --------------------

def make_flowers_loaders(
    *,
    root: str = "./data",
    batch_size: int,
    image_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = True,
) -> DataLoader: #only training

    # Flowers102 images have variable resolution; resize is REQUIRED for batching.
    tfm = _tfm_to_minus1_1(image_size=image_size)

    # Flowers102 expects a directory root for its own files; put it under {root}/flowers
    flowers_root = os.path.join(root, "flowers")
    ds_train = tv_datasets.Flowers102(
        root=flowers_root, split="train", download=True, transform=tfm
    )
    ds_val = tv_datasets.Flowers102(
        root=flowers_root, split="val", download=True, transform=tfm
    )
    ds_test = tv_datasets.Flowers102(
        root=flowers_root, split="test", download=True, transform=tfm
    )
    train_ds = ConcatDataset([ds_train, ds_val, ds_test])
    print(f"[Flowers102] total training images: {len(train_ds)}")  # should be 8189

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return train_dl


# ---- other utilities ----

def infinite_loader(dl):
    """Yield batches forever; each pass through dl is one epoch."""
    while True:
        for batch in dl:
            yield batch

@torch.no_grad()
def dump_real_png_subset(ds, out_dir, n=50000):
    os.makedirs(out_dir, exist_ok=True)
    n = min(n, len(ds))

    for i in tqdm(range(n), desc="Dump real PNG subset", ncols=88):
        x, _ = ds[i]

        # Accept either CHW float tensor in [-1,1] (your ImageNet32 / training pipelines)
        # or a PIL image (if you ever call this on a raw torchvision dataset).
        if torch.is_tensor(x):
            # x: CHW float, assumed in [-1,1]
            x = (x.clamp(-1, 1) * 0.5 + 0.5)         # -> [0,1]
            x = (x * 255.0).round().to(torch.uint8)  # uint8
            x = x.permute(1, 2, 0).cpu().numpy()     # HWC
            img = Image.fromarray(x)
        else:
            # PIL image
            img = x

        img.save(os.path.join(out_dir, f"{i:06d}.png"))
        

