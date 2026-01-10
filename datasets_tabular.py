# this file follows the setup at https://github.com/francois-rozet/uci-datasets to preprocess the 
# tabular datasets and create train/val/test splits
# 
# Unified tabular dataset loading + preprocessing for:
#   - bsds300
#   - gas
#   - miniboone
#   - power
#
# Expected on-disk layout under data_root (default "./data"):
#   BSDS300/BSDS300.hdf5
#   gas/ethylene_CO.pickle
#   miniboone/data.npy
#   power/data.npy
#

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

# -----------------------------
# Containers
# -----------------------------
@dataclass
class Split:
    x: np.ndarray

    @property
    def N(self) -> int:
        return int(self.x.shape[0])


@dataclass
class TabularDataset:
    trn: Split
    val: Split
    tst: Split
    n_dims: int
    meta: Dict[str, Any]



# Path handling
def _as_root(data_root: Optional[str | Path]) -> Path:
    if data_root is None:
        return Path("data").resolve()
    return Path(data_root).expanduser().resolve()


def _need_file(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")


# data preprocessing
def _bsds300(root: Path) -> TabularDataset:
    # original preprocess code:
    #   f = h5py.File(datasets.root + 'BSDS300/BSDS300.hdf5', 'r')
    #   trn = f['train']; val = f['validation']; tst = f['test']
    p = root / "BSDS300" / "BSDS300.hdf5"
    _need_file(p)
    try:
        import h5py
    except ImportError as e:
        raise ImportError("BSDS300 requires h5py. Install via: pip install h5py") from e

    with h5py.File(str(p), "r") as f:
        trn = np.array(f["train"], dtype=np.float32)
        val = np.array(f["validation"], dtype=np.float32)
        tst = np.array(f["test"], dtype=np.float32)

    n_dims = int(trn.shape[1])
    image_size = [int(np.sqrt(n_dims + 1))] * 2  # kept for parity with original preprocess code

    return TabularDataset(
        trn=Split(trn),
        val=Split(val),
        tst=Split(tst),
        n_dims=n_dims,
        meta={"name": "bsds300", "image_size": image_size},
    )


def _gas(root: Path) -> TabularDataset:
    # original preprocess code:
    # load pickle -> drop Meth/Eth/Time -> iteratively drop cols with corr > 0.98 -> standardize
    # then split: test=last 10%; val=last 10% of remaining; train=rest.
    p = root / "gas" / "ethylene_CO.pickle"
    _need_file(p)
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("GAS requires pandas. Install via: pip install pandas") from e

    data = pd.read_pickle(str(p))
    data.drop("Meth", axis=1, inplace=True)
    data.drop("Eth", axis=1, inplace=True)
    data.drop("Time", axis=1, inplace=True)

    def get_correlation_numbers(df):
        C = df.corr()
        A = C > 0.98
        B = A.values.sum(axis=1)
        return B

    B = get_correlation_numbers(data)
    while np.any(B > 1):
        col_to_remove = np.where(B > 1)[0][0]
        col_name = data.columns[col_to_remove]
        data.drop(col_name, axis=1, inplace=True)
        B = get_correlation_numbers(data)

    data = (data - data.mean()) / data.std()
    arr = data.values.astype(np.float32)

    N_test = int(0.1 * arr.shape[0])
    tst = arr[-N_test:]
    trnval = arr[0:-N_test]
    N_validate = int(0.1 * trnval.shape[0])
    val = trnval[-N_validate:]
    trn = trnval[0:-N_validate]

    n_dims = int(trn.shape[1])
    return TabularDataset(
        trn=Split(trn),
        val=Split(val),
        tst=Split(tst),
        n_dims=n_dims,
        meta={"name": "gas"},
    )


def _miniboone(root: Path) -> TabularDataset:
    # original preprocess code:
    #   data = np.load(...)
    #   split (no shuffle): test last 10%, val last 10% of remaining, train rest
    #   normalize using mu/std computed from stack(train, val)
    p = root / "miniboone" / "data.npy"
    _need_file(p)
    data = np.load(str(p))

    N_test = int(0.1 * data.shape[0])
    tst = data[-N_test:]
    data2 = data[0:-N_test]
    N_validate = int(0.1 * data2.shape[0])
    val = data2[-N_validate:]
    trn = data2[0:-N_validate]

    trnval = np.vstack((trn, val))
    mu = trnval.mean(axis=0)
    s = trnval.std(axis=0)
    trn = ((trn - mu) / s).astype(np.float32)
    val = ((val - mu) / s).astype(np.float32)
    tst = ((tst - mu) / s).astype(np.float32)

    n_dims = int(trn.shape[1])
    return TabularDataset(
        trn=Split(trn),
        val=Split(val),
        tst=Split(tst),
        n_dims=n_dims,
        meta={"name": "miniboone"},
    )


def _power(root: Path) -> TabularDataset:
    # original preprocess code:
    #   rng = RandomState(42); data = np.load(...); rng.shuffle(data)
    #   delete cols 3 then 1
    #   add uniform noise: gap(0.001), voltage(0.01), sm(1.0), time(0)
    #   split last 10%, then val last 10% of remaining
    #   normalize using mu/std on stack(train, val)
    p = root / "power" / "data.npy"
    _need_file(p)
    data = np.load(str(p))

    rng = np.random.RandomState(42)
    rng.shuffle(data)
    N = data.shape[0]

    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)

    voltage_noise = 0.01 * rng.rand(N, 1)
    gap_noise = 0.001 * rng.rand(N, 1)
    sm_noise = rng.rand(N, 3)
    time_noise = np.zeros((N, 1))
    noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
    data = data + noise

    N_test = int(0.1 * data.shape[0])
    tst = data[-N_test:]
    data2 = data[0:-N_test]
    N_validate = int(0.1 * data2.shape[0])
    val = data2[-N_validate:]
    trn = data2[0:-N_validate]

    trnval = np.vstack((trn, val))
    mu = trnval.mean(axis=0)
    s = trnval.std(axis=0)
    trn = ((trn - mu) / s).astype(np.float32)
    val = ((val - mu) / s).astype(np.float32)
    tst = ((tst - mu) / s).astype(np.float32)

    n_dims = int(trn.shape[1])
    return TabularDataset(
        trn=Split(trn),
        val=Split(val),
        tst=Split(tst),
        n_dims=n_dims,
        meta={"name": "power"},
    )

# -----------------------------
# Public API
# -----------------------------
def get_tabular_dataset(
    name: str,
    *,
    data_root: Optional[str | Path] = None,
) -> TabularDataset:
    """
    Create a TabularDataset with the *exact* preprocessing & split logic from the original preprocess code.

    Args:
        name: one of {"bsds300","gas","miniboone","power"} (case-insensitive)
        data_root: root folder that contains the dataset task subfolders. Default: "./data".

    Returns:
        TabularDataset with numpy arrays in ds.trn.x, ds.val.x, ds.tst.x
    """
    key = name.strip().lower()
    root = _as_root(data_root)

    if key == "bsds300":
        return _bsds300(root)
    if key == "gas":
        return _gas(root)
    if key == "miniboone":
        return _miniboone(root)
    if key == "power":
        return _power(root)

    raise ValueError(f"Unknown tabular dataset '{name}'. Supported: bsds300, gas, miniboone, power")

# DataLoader helpers 
def make_tabular_loaders(
    name: str,
    *,
    data_root: Optional[str | Path] = None,
    batch_size: int = 512,
    test_batch_size: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    persistent_workers: bool = False,
):
    """Create PyTorch DataLoaders for tabular datasets (train + val+ test).
    This keeps **all data on CPU** and only moves a batch to GPU inside your training loop.
    Returns:
        train_dl, val_dl, test_dl, info_dict
    """
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise ImportError("make_tabular_loaders requires PyTorch installed.") from e

    ds = get_tabular_dataset(name, data_root=data_root)

    xtr = torch.from_numpy(ds.trn.x)  # CPU
    xva = torch.from_numpy(ds.val.x)  # CPU
    xte = torch.from_numpy(ds.tst.x)  # CPU

    if test_batch_size is None:
        test_batch_size = batch_size

    train_ds = TensorDataset(xtr)
    val_ds = TensorDataset(xva)
    test_ds = TensorDataset(xte)

    # Important: only training split is shuffled.
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size= batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    info = {
        "name": ds.meta.get("name", name.strip().lower()),
        "n_dims": ds.n_dims,
        "n_train": ds.trn.N,
        "n_val": ds.val.N,
        "n_test": ds.tst.N,
    }
    return train_dl, val_dl, test_dl, info
