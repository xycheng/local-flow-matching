import math
import os
import re
import argparse
import yaml
from copy import deepcopy
from collections import OrderedDict
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import torchdiffeq as tdeq
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import chi2

# -------------------------
# utils 
# -------------------------

def unwrap_x(batch):
    if isinstance(batch, (list, tuple)) and len(batch) >= 1:
        return batch[0]
    return batch

def infinite_loader(dl: DataLoader):
    while True:
        for b in dl:
            yield b

def make_tensor_loader(x_cpu: torch.Tensor, batch_size: int, *,
                       shuffle: bool = True, drop_last: bool = True,
                       num_workers: int = 0, pin_memory: bool = True) -> DataLoader:
    ds = TensorDataset(x_cpu)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

# optional: keep whole split tensors on device (faster for tiny nets / huge batches)
def get_ds_tensor(dl: DataLoader) -> torch.Tensor:
    """Extract the single tensor from a TensorDataset-backed loader."""
    ds = dl.dataset
    if hasattr(ds, "tensors") and len(ds.tensors) >= 1:
        return ds.tensors[0]
    xs = []
    for b in DataLoader(ds, batch_size=65536, shuffle=False, drop_last=False, num_workers=0):
        xs.append(unwrap_x(b))
    return torch.cat(xs, dim=0)

def infinite_tensor_batches(x_dev: torch.Tensor, batch_size: int, *,
                           shuffle: bool = True, drop_last: bool = True):
    """Yield device batches by indexing a device-resident tensor."""
    n = x_dev.shape[0]
    while True:
        if shuffle:
            perm = torch.randperm(n, device=x_dev.device)
        else:
            perm = torch.arange(n, device=x_dev.device)
        if drop_last:
            n_use = (n // batch_size) * batch_size
            perm = perm[:n_use]
        for i in range(0, perm.numel(), batch_size):
            idx = perm[i:i+batch_size]
            if idx.numel() < batch_size and drop_last:
                continue
            yield x_dev.index_select(0, idx)

def make_subset_loader(dl: DataLoader, n_eval: int,
                       batch_size: int | None = None,
                       num_workers: int | None = None,
                       pin_memory: bool | None = None) -> DataLoader:
    ds = dl.dataset
    n = len(ds)
    k = min(int(n_eval), n)
    idx = torch.randperm(n)[:k].tolist()
    return DataLoader(
        Subset(ds, idx),
        batch_size=batch_size if batch_size is not None else dl.batch_size,
        shuffle=False,
        num_workers=num_workers if num_workers is not None else dl.num_workers,
        pin_memory=pin_memory if pin_memory is not None else dl.pin_memory,
        drop_last=False,
    )

# -------------------------
# data (2D "image mask" dataset; stays on CPU)
# -------------------------

def _gen_data_from_img_mask(image_mask: np.ndarray, n: int) -> np.ndarray:
    """
    FFJORD-style sampling from a grayscale image mask.
    Returns (n,2) float64 array in roughly [-4,4]^2.
    """
    img = image_mask
    h, w = img.shape
    xx = np.linspace(-4.0, 4.0, w)
    yy = np.linspace(-4.0, 4.0, h)
    xx, yy = np.meshgrid(xx, yy)
    means = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)  # (h*w,2)

    img = img.max() - img
    probs = img.reshape(-1).astype(np.float64)
    probs = probs / (probs.sum() + 1e-12)

    std = np.array([8.0 / w / 2.0, 8.0 / h / 2.0], dtype=np.float64)

    inds = np.random.choice(int(probs.shape[0]), int(n), p=probs)
    m = means[inds]
    samples = np.random.randn(*m.shape) * std + m
    return samples

def load_2d_image_dataset(dataname: str, *, data_root: str, ntr: int, nte: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expects '{data_root}/img_{dataname}.png'. Returns (train_cpu, test_cpu), both float32 CPU tensors.
    """
    path = os.path.join(data_root, f"img_{dataname}.png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"2D dataset image not found: {path}")

    img = Image.open(path).rotate(180).transpose(0).convert("L")
    mask = np.array(img)

    rng_state = np.random.get_state()
    np.random.seed(seed)

    data = _gen_data_from_img_mask(mask, ntr + nte)

    np.random.set_state(rng_state)

    xtr = torch.from_numpy(data[:ntr]).float().cpu()
    xte = torch.from_numpy(data[ntr:]).float().cpu()
    return xtr, xte

def OU_at_t(x: torch.Tensor, t: float, z: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Ornstein–Uhlenbeck forward step: x -> shrink*x + sqrt(1-shrink^2)*z, with shrink = exp(-t).
    All tensors on same device as x.
    """
    if z is None:
        z = torch.randn_like(x)
    shrink = math.exp(-float(t))
    sigma = math.sqrt(max(0.0, 1.0 - shrink ** 2))
    return shrink * x + sigma * z

# -------------------------
# model (CNF) + loss 
# -------------------------

def build_mlp(in_features: int, hidden_features: List[int], *, cat_t: str = "first"):
    if cat_t not in ("first", "all"):
        raise ValueError(f"cat_t must be 'first' or 'all', got {cat_t}")
    layers = nn.ModuleList()
    for i, (a, b) in enumerate(zip([in_features] + hidden_features, hidden_features + [in_features])):
        add_t = (cat_t == "all") or (cat_t == "first" and i == 0)
        layers.append(nn.Linear(a + (1 if add_t else 0), b))
    return layers

class CNF(nn.Module):
    def __init__(self, data_dim: int, hidden_features: List[int], *,
                 act: str = "elu", cat_t: str = "first", ode_cfg: Optional[dict] = None):
        super().__init__()
        self.data_dim = int(data_dim)
        self.net = build_mlp(self.data_dim, hidden_features, cat_t=cat_t)
        self.cat_t = cat_t
        act = act.lower()
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "softplus":
            self.act = nn.Softplus(beta=20)
        else:
            self.act = nn.ELU()

        self.ode_cfg = ode_cfg or {"method": "dopri5", "tol": 1e-5, "num_steps": 100}

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        if t.ndim == 2:
            tt = t
        else:
            tt = torch.ones_like(x[:, :1]) * t

        h = x
        for i, layer in enumerate(self.net):
            add_t = (self.cat_t == "all") or (self.cat_t == "first" and i == 0)
            if add_t:
                h = torch.cat([tt, h], dim=1)
            h = layer(h)
            if i < len(self.net) - 1:
                h = self.act(h)
        return h

    @torch.no_grad()
    def front_or_back(self, z: torch.Tensor, *, towards_one: bool = False):
        self.eval()
        first, second = (0.0, 1.0) if towards_one else (1.0, 0.0)

        method = str(self.ode_cfg.get("method", "dopri5")).lower()
        if method == "dopri5":
            tol = float(self.ode_cfg.get("tol", 1e-5))
            tspan = torch.tensor([first, second], device=z.device, dtype=z.dtype)
            return tdeq.odeint(self, z, tspan, method="dopri5", rtol=tol, atol=tol)

        if method in ("rk4", "euler"):
            num_steps = int(self.ode_cfg.get("num_steps", 50))
            tspan = torch.linspace(first, second, num_steps, device=z.device, dtype=z.dtype)
            return tdeq.odeint(self, z, tspan, method=method)

        raise ValueError(f"Unknown ODE method: {method}")

    def log_prob(self, x: torch.Tensor, *, return_full: bool = False):
        """
        Exact trace via autograd basis (fine for D=2).
        """
        self.eval()
        device = x.device
        D = x.shape[-1]

        I = torch.eye(D, device=device, dtype=x.dtype)
        I = I.expand(x.shape + (D,)).movedim(-1, 0)  # (D, B, D)

        def augmented(t, x_ladj):
            xt, ladj = x_ladj
            with torch.enable_grad():
                xt = xt.requires_grad_()
                dx = self(t, xt)
            jacobian = torch.autograd.grad(dx, xt, I, is_grads_batched=True, create_graph=True)[0]
            trace = torch.einsum("i...i", jacobian)
            return dx, trace

        ladj0 = torch.zeros(x.shape[0], device=device, dtype=x.dtype)

        method = str(self.ode_cfg.get("method", "dopri5")).lower()
        if method == "dopri5":
            tol = float(self.ode_cfg.get("tol", 1e-5))
            tspan = torch.tensor([0.0, 1.0], device=device, dtype=x.dtype)
            with torch.no_grad():
                z, ladj = tdeq.odeint(augmented, (x, ladj0), tspan, method="dopri5", rtol=tol, atol=tol)
        else:
            # for fixed-step methods, reuse num_steps
            num_steps = int(self.ode_cfg.get("num_steps", 50))
            tspan = torch.linspace(0.0, 1.0, num_steps, device=device, dtype=x.dtype)
            with torch.no_grad():
                z, ladj = tdeq.odeint(augmented, (x, ladj0), tspan, method=method)

        z = z[-1]
        ladj = ladj[-1]

        if not return_full:
            return ladj

        log_qz = Normal(0.0, z.new_tensor(1.0)).log_prob(z).sum(dim=-1)
        return log_qz + ladj

class InterFlowLoss(nn.Module):
    """
    Flow Matching loss with selectable interpolation:
      - interp="linear": I_t = x0 + t (x1-x0), dI_t = x1-x0
      - interp="trig":   I_t = cos(pi t/2)x0 + sin(pi t/2)x1
    """
    def __init__(self, v: nn.Module, beta_cfg: Optional[dict] = None, interp: str = "linear"):
        super().__init__()
        self.v = v
        self.beta_cfg = beta_cfg
        interp = interp.lower()
        if interp not in ("linear", "trig"):
            raise ValueError(f"Unknown interp='{interp}'. Use 'linear' or 'trig'.")
        self.interp = interp

    def sample_t(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.beta_cfg is not None and self.beta_cfg.get("enabled", False):
            alpha = float(self.beta_cfg["alpha"])
            beta = float(self.beta_cfg["beta"])
            t = torch.distributions.Beta(alpha, beta).sample((B, 1)).to(device=device, dtype=dtype)
        else:
            t = torch.rand(B, 1, device=device, dtype=dtype)
        return t

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        t = self.sample_t(B, x0.device, x0.dtype)  # (B,1)

        if self.interp == "linear":
            dIt = x1 - x0
            It = x0 + t * dIt
        else:
            c = torch.cos((math.pi / 2.0) * t)
            s = torch.sin((math.pi / 2.0) * t)
            It = c * x0 + s * x1
            dIt = (math.pi / 2.0) * (c * x1 - s * x0)

        vout = self.v(t, It)
        return (vout - dIt).pow(2).mean()

# -------------------------
# push + eval (NLL)
# -------------------------

@torch.no_grad()
def push_loader_by_current(flow: CNF, dl: DataLoader, *, device: torch.device) -> torch.Tensor:
    flow.eval()
    outs = []
    for batch in tqdm(dl, desc="Pushforward all", ncols=88):
        x = unwrap_x(batch).to(device, dtype=torch.float32, non_blocking=True)
        y = flow.front_or_back(x, towards_one=True)[-1].detach().cpu()
        outs.append(y)
    return torch.cat(outs, dim=0)

@torch.no_grad()
def eval_nll_loader(test_dl: DataLoader, prev_blocks: List[CNF], cur_block: CNF, *,
                    device: torch.device) -> float:
    cur_block.eval()
    for b in prev_blocks:
        b.eval()

    total_nll = 0.0
    total_n = 0

    for (x_cpu,) in tqdm(test_dl, desc="Evaluating NLL", leave=False, ncols=88):
        bs = x_cpu.shape[0]
        x = x_cpu.to(device, dtype=torch.float32, non_blocking=True)

        log_p_prev = 0.0
        for blk in prev_blocks:
            log_p_prev += blk.log_prob(x, return_full=False).mean().item()
            x = blk.front_or_back(x, towards_one=True)[-1]

        lp = cur_block.log_prob(x, return_full=True).mean().item()
        lp = lp + log_p_prev

        total_nll += (-lp) * bs
        total_n += bs

    return float(total_nll / max(total_n, 1))

@torch.no_grad()
def push_tensor_by_current(flow: CNF, x_dev: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    """Pushforward a whole device tensor through `flow` in chunks; returns device tensor."""
    flow.eval()
    outs = []
    n = x_dev.shape[0]
    for i in tqdm(range(0, n, batch_size), desc="Pushforward all (device)", ncols=88):
        xb = x_dev[i:i+batch_size]
        y = flow.front_or_back(xb, towards_one=True)[-1]
        outs.append(y.detach())
    return torch.cat(outs, dim=0)

@torch.no_grad()
def eval_nll_tensor(x_dev: torch.Tensor, prev_blocks: List[CNF], cur_block: CNF, *,
                    batch_size: int) -> float:
    """Evaluate NLL over a device tensor, batching internally."""
    cur_block.eval()
    for b in prev_blocks:
        b.eval()

    total_nll = 0.0
    total_n = 0
    n = x_dev.shape[0]
    for i in tqdm(range(0, n, batch_size), desc="Evaluating NLL (device)", leave=False, ncols=88):
        x = x_dev[i:i+batch_size]

        log_p_prev = 0.0
        for blk in prev_blocks:
            log_p_prev += blk.log_prob(x, return_full=False).mean().item()
            x = blk.front_or_back(x, towards_one=True)[-1]

        lp = cur_block.log_prob(x, return_full=True).mean().item() + log_p_prev
        nll_batch = -lp
        bs = x.shape[0]
        total_nll += nll_batch * bs
        total_n += bs

    return float(total_nll / max(total_n, 1))

# -------------------------
# other training related
# -------------------------


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

def find_block_ckpt(master_dir: str, block_id: int, seed: int):
    """
    Priority:
      1) state_dict_block_{block_id}_seed{seed}.pt
      2) state_dict_block_{block_id}_seed{seed}_latest.pt
      3) state_dict_block_{block_id}_seed{seed}_*.pt  (largest trailing number)
    """
    canonical = os.path.join(master_dir, f"state_dict_block_{block_id}_seed{seed}.pt")
    if os.path.exists(canonical):
        return canonical

    latest = os.path.join(master_dir, f"state_dict_block_{block_id}_seed{seed}_latest.pt")
    if os.path.exists(latest):
        return latest

    pattern = re.compile(rf"state_dict_block_{block_id}_seed{seed}_(\d+)\.pt")
    best_fname, best_step = None, -1
    for fname in os.listdir(master_dir):
        m = pattern.fullmatch(fname)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step, best_fname = step, fname
    if best_fname is None:
        return None
    return os.path.join(master_dir, best_fname)


# -------------------------
# visualization (kept lightweight; uses CPU tensors, minibatches)
# -------------------------

def plot_true_ou(
    xte_cpu: torch.Tensor,
    hks: List[Optional[float]],
    *,
    master_dir: str,
    s: float,
    max_points: int = 5000,
):
    """
    Visualization of the *true* OU forward corruption chain (not learned).
    Saves: {master_dir}/True_OU.png
    """
    grid_size = 4
    fsize = 26

    x = xte_cpu[:max_points].clone()
    data_ls = [x]

    for i, hk in enumerate(hks):
        is_last = (i == len(hks) - 1)
        if hk is None or is_last:
            data_ls.append(torch.randn_like(x))
        else:
            data_ls.append(OU_at_t(data_ls[-1], hk))

    data = torch.stack(data_ls).numpy()  # (T, N, 2)
    num_cols = data.shape[0]

    fig, ax = plt.subplots(1, num_cols, figsize=(num_cols*4, 4), sharey=True, sharex=True)
    for i in range(num_cols):
        ax[i].scatter(data[i, :, 0], data[i, :, 1], s=s)
        if i == 0:
            title = 'Raw data'
        elif i < num_cols-1:
            title = f't={np.cumsum(hks[:i])[-1].item():.3f}'
        else:
            title = 'Standard normal'
        fsize = 26
        ax[i].set_title(title, fontsize=fsize)
    fig.savefig(os.path.join(master_dir, 'True_OU.png'), dpi=100)
    plt.close()

def plot_loss_curve(loss_ls: List[float], *, out_dir: str, suffix: str = "latest"):
    """loss plot, saved as Loss_<suffix>.png (overwritten)."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(loss_ls)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"Loss_{suffix}.png"), dpi=120)
    plt.close()

def scatter_save(path: str, pts: np.ndarray, title: str, *, s: float = 0.25):
    plt.figure(figsize=(4, 4))
    plt.scatter(pts[:, 0], pts[:, 1], s=s)
    plt.title(title, fontsize=16)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()


@torch.no_grad()
def _compose_forward(blocks: List[CNF], x: torch.Tensor) -> Tuple[List[np.ndarray], torch.Tensor]:
    snaps = [x.detach().cpu().numpy()]
    for blk in blocks:
        x = blk.front_or_back(x, towards_one=True)[-1]
        snaps.append(x.detach().cpu().numpy())
    return snaps, x

@torch.no_grad()
def _compose_backward(blocks: List[CNF], y: torch.Tensor) -> Tuple[List[np.ndarray], torch.Tensor]:
    snaps = [y.detach().cpu().numpy()]
    for blk in blocks[::-1]:
        y = blk.front_or_back(y, towards_one=False)[-1]
        snaps.append(y.detach().cpu().numpy())
    return snaps, y


@torch.no_grad()
def viz_block_samples(
    xte_cpu: torch.Tensor,
    *,
    yte_cpu: torch.Tensor,
    prev_blocks: List[CNF],
    cur_block: CNF,
    hk: float | None,
    device: torch.device,
    out_dir: str,
    tag: str,
    n_plot: int = 5000,
    s: float = 0.00025,
):
    """
    Visualization, saved directly under out_dir:
      - Gen_batch_<tag>.png   : Xhat | Yhat | Ytrue
      - Traj_batch_<tag>.png  : trajectory frames t=... then Y

    This function is called with tag='latest' at vis_freq and tag='final' at end of block.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Infer current block index l from how many prev blocks exist: l = len(prev_blocks)+1
    l = len(prev_blocks) + 1

    # Get total blocks L if available (for correct "last block = Gaussian" ytrue); otherwise fall back to hk is None
    # Prefer global L if present
    L_local = globals().get("L", None)

    # ---- slice + move points ----
    xte_raw = xte_cpu[:n_plot].clone()
    x_in = xte_raw.to(device=device, dtype=torch.float32)
    y_noise = yte_cpu[:n_plot].to(device=device, dtype=torch.float32)

    # ---- forward through previous blocks to current input ----
    if l > 1:
        ypast_list, x_cur = _compose_forward(prev_blocks, x_in)
        # prev_blocks length = l-1; ypast_list includes input => length l
        ypast = np.stack(ypast_list, axis=0)  # (l, N, 2)
    else:
        x_cur = x_in

    # ---- current forward: Yhat ----
    yhat_end = cur_block.front_or_back(x_cur, towards_one=True)[-1]
    yhat = yhat_end.detach().cpu().numpy()

    # ---- current inverse + previous inverse: Xhat ----
    xhat_after_cur = cur_block.front_or_back(y_noise, towards_one=False)[-1]
    if l > 1:
        xpast_list, xhat_final_t = _compose_backward(prev_blocks, xhat_after_cur)
        xpast = np.stack(xpast_list[1:], axis=0)  # (l-1, N, 2) after each inverse
        xhat_final = xhat_final_t.detach().cpu().numpy()
    else:
        xhat_final = xhat_after_cur.detach().cpu().numpy()

    # ---- Ytrue at this block (last block -> Gaussian) ----
    is_last_block = False
    if L_local is not None:
        is_last_block = (l == int(L_local) + 1)
    else:
        # if we don't know L, use hk None as "Gaussian"
        is_last_block = (hk is None)

    if is_last_block or hk is None:
        ytrue = torch.randn_like(x_cur).detach().cpu().numpy()
    else:
        ytrue = OU_at_t(x_cur, hk).detach().cpu().numpy()

    # ---- Blocks arrays  ----
    if l > 1:
        yte_np = y_noise.detach().cpu().numpy().reshape(1, *xte_raw.shape)  # (1,N,2)
        xhat_np = xhat_after_cur.detach().cpu().numpy().reshape(1, *xte_raw.shape)  # (1,N,2)
        xte_np = xte_raw.detach().cpu().numpy().reshape(1, *xte_raw.shape)  # (1,N,2)

        # X row: [Y, Xhat(current), xpast..., X]
        xhat_blocks = np.concatenate([yte_np, xhat_np, xpast, xte_np], axis=0)  # (l+2,N,2)

        # Y row: [X, after prev1, ..., after prev(l-1), Yhat, Y]
        yhat_blocks = np.concatenate([ypast, yhat.reshape(1, *xte_raw.shape), yte_np], axis=0)  # (l+2,N,2)
    else:
        # l == 1: use endpoints, keep same shapes
        xhat_blocks = np.stack([xhat_after_cur.detach().cpu().numpy(), xhat_final, xte_raw.detach().cpu().numpy()], axis=0)
        yhat_blocks = np.stack([x_cur.detach().cpu().numpy(), yhat, y_noise.detach().cpu().numpy()], axis=0)

    # ---- Trajectory through current block on a fixed grid of num_steps, then append ytrue ----
    ode_cfg = getattr(cur_block, "ode_cfg", {})
    method = str(ode_cfg.get("method", "dopri5")).lower()
    num_steps = int(ode_cfg.get("num_steps", 5))  # default 5 frames
    tspan = torch.linspace(0.0, 1.0, num_steps, device=device, dtype=torch.float32)

    if method == "dopri5":
        tol = float(ode_cfg.get("tol", 1e-5))
        yhat_traj = tdeq.odeint(cur_block, x_cur, tspan, method="dopri5", rtol=tol, atol=tol)
    else:
        yhat_traj = tdeq.odeint(cur_block, x_cur, tspan, method=method)

    yhat_traj = yhat_traj.detach().cpu().numpy()  # (num_steps,N,2)
    yhat_traj = np.concatenate([yhat_traj, ytrue.reshape(1, *xte_raw.shape)], axis=0)  # (num_steps+1,N,2)

    # ---- plotting params ----
    grid_size = 4
    fsize = 26

    # Gen figure 
    fig, ax = plt.subplots(1, 3, figsize=(3 * grid_size, 4))
    ax[0].scatter(xhat_final[:, 0], xhat_final[:, 1], s=s)
    ax[0].set_title("Xhat", fontsize=fsize)
    ax[1].scatter(yhat[:, 0], yhat[:, 1], s=s)
    ax[1].set_title("Yhat", fontsize=fsize)
    ax[2].scatter(ytrue[:, 0], ytrue[:, 1], s=s)
    ax[2].set_title(f"Ytrue, Block {l}", fontsize=fsize)
    fig.savefig(os.path.join(out_dir, f"Gen_batch_{tag}.png"), dpi=100)
    plt.close(fig)

    #  Trajectory figure 
    fig, axs = plt.subplots(1, num_steps + 1, figsize=(grid_size * (num_steps + 1), grid_size), sharex="row", sharey="row")
    weights = np.linspace(0.0, 1.0, num_steps)
    for j in range(num_steps + 1):
        axj = axs[j]
        axj.scatter(yhat_traj[j, :, 0], yhat_traj[j, :, 1], s=s)
        if j == num_steps:
            axj.set_title("Y", fontsize=fsize)
        else:
            axj.set_title(f"t={weights[j]:.2f}", fontsize=fsize)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"Traj_batch_{tag}.png"), dpi=100)
    plt.close(fig)

@torch.no_grad()
def viz_confidence_regions(*, prev_blocks: List[CNF], cur_block: CNF,
                           device: torch.device, out_path: str,
                           alphas: List[float] = [0.05, 0.2, 0.4, 0.6, 0.8],
                           n_angles: int = 4000):
    """
    Draw chi-square confidence regions in latent z-space and map back to x-space.
    (Uses inverse map of composed flow.)
    """
    angles = torch.linspace(0.0, 2.0 * math.pi, n_angles, device=device)
    dim = 2

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    for alpha in alphas:
        radius = float(math.sqrt(chi2.ppf(1.0 - alpha, df=dim)))
        z = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1).float()

        x = cur_block.front_or_back(z, towards_one=False)[-1]
        for blk in prev_blocks[::-1]:
            x = blk.front_or_back(x, towards_one=False)[-1]
        pts = x.detach().cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=0.15, label=f"1-α={1-alpha:.2f}")

    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=10)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)

@torch.no_grad()
def sample_final(n: int,*,flow_ls: List[CNF],device: torch.device) -> np.ndarray:
    z = torch.randn(n, 2, device=device, dtype=torch.float32)
    x = flow_ls[-1].front_or_back(z, towards_one=False)[-1]
    for blk in flow_ls[:-1][::-1]:
        x = blk.front_or_back(x, towards_one=False)[-1]
    return x.detach().cpu().numpy()


def _coerce_hk(h):
    if h is None: return None
    if isinstance(h, str) and h.strip().lower() in ("none", "null", "~"):
        return None
    return float(h)



# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="2D LFM training (refactored to main_tabular.py style)")
    parser.add_argument("--hyper_param_config", type=str, default="config/2d_tree.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--all_on_device", action="store_true", help="Keep train/test tensors on device and batch by device indexing (faster if data fits GPU)")
    args = parser.parse_args()

    with open(args.hyper_param_config, "r") as f:
        cfg = yaml.safe_load(f)
    print(yaml.dump(cfg, default_flow_style=False))

    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[INFO] Random seed set to {seed}")

    device_cfg = args.device if args.device is not None else cfg.get("device")
    if device_cfg is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
        if device.type == "cuda" and device.index is not None:
            torch.cuda.set_device(device.index)

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    base_dir = cfg.get("base_dir", "results_2d")
    save_dir = cfg.get("save_dir", "debug_run")
    master_dir = os.path.join(base_dir, save_dir)
    os.makedirs(master_dir, exist_ok=True)
    with open(os.path.join(master_dir, "config.txt"), "w") as f:
        f.write(yaml.dump(cfg, default_flow_style=False))

    # ---- data (CPU) ----
    data_root = cfg["data"].get("data_root", "./data")
    dataname = cfg["data"]["dataname"]
    ntr = int(cfg["data"]["ntr"])
    nte = int(cfg["data"]["nte"])

    xtr_cpu, xte_cpu = load_2d_image_dataset(dataname, data_root=data_root, ntr=ntr, nte=nte, seed=seed)
    yte_cpu = torch.randn_like(xte_cpu)

    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["data"].get("num_workers", 0))
    pin_memory = bool(cfg["data"].get("pin_memory", True))
    test_batch_size = int(cfg.get("eval", {}).get("test_batch_size", batch_size))

    train_dl = DataLoader(
        TensorDataset(xtr_cpu),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_dl = DataLoader(
        TensorDataset(xte_cpu),
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    # optional: keep whole splits on device (faster when data fits)
    all_on_device = bool(args.all_on_device or cfg.get("data", {}).get("all_on_device", False))
    if all_on_device:
        xtr_dev = xtr_cpu.to(device=device, dtype=torch.float32)
        xte_dev = xte_cpu.to(device=device, dtype=torch.float32)
        yte_dev = yte_cpu.to(device=device, dtype=torch.float32)
        train_x_dev = xtr_dev  # updated by pushforward across blocks
        print(f"[all_on_device] train={tuple(train_x_dev.shape)} test={tuple(xte_dev.shape)} on {device}")
    else:
        xtr_dev = None
        xte_dev = None
        yte_dev = None
        train_x_dev = None


    # ---- model & training cfg ----
    L = int(cfg["model"]["L"])  # blocks are 1..L+1
    hks = cfg["model"]["hks"]
    hks = [_coerce_hk(h) for h in hks]
    assert len(hks) == L + 1, "Expected hks length L+1 (one per block)."

    hidden_dim = int(cfg["model"]["hidden_dim"])
    num_hidden = int(cfg["model"]["num_hidden"])
    act = str(cfg["model"].get("act", "elu"))
    cat_t = str(cfg["model"].get("cat_t", "first"))
    interp = str(cfg["model"].get("interp", "linear")).lower()
    coupling = str(cfg["model"].get("coupling", "dependent")).lower()
    ode_cfg = cfg.get("ode", {"method": "dopri5", "tol": 1e-5, "num_steps": 100})

    beta_cfg = None
    if cfg["model"].get("beta_sample_t", False):
        beta_cfg = {"enabled": True, "alpha": cfg["model"]["alpha"], "beta": cfg["model"]["beta"]}
        print(f"[t-sampling] Beta({beta_cfg['alpha']},{beta_cfg['beta']}) enabled")

    max_batch_ls = cfg["training"]["max_batch"]
    resume = bool(cfg["training"].get("resume", False))
    eval_only = bool(cfg["training"].get("eval_only", False))
    warm_start = bool(cfg["training"].get("warm_start", False))

    use_grad_clip = bool(cfg["training"].get("use_grad_clip", True))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    use_ema = bool(cfg["training"].get("use_ema", True))
    load_ema = bool(cfg["training"].get("load_ema", True)) and use_ema
    ema_decay = float(cfg["training"].get("ema_decay", 0.9999)) if use_ema else 0.0

    viz_freq = int(cfg.get("visualize", {}).get("viz_freq", 2000))
    batch_push_size = int(cfg["training"].get("batch_push_size", 8192))

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ---- quick sanity plot of the true OU chain (CPU) ----
    plot_true_ou(xte_cpu, hks, master_dir=master_dir, s=float(cfg.get("visualize", {}).get("scatter_s", 0.25)),
                 max_points=int(cfg.get("visualize", {}).get("n_plot", 5000)))

    # ---- eval-only path ----
    if eval_only:
        flow_ls: List[CNF] = []
        for l_now in range(1, L + 2):
            flow = CNF(2, [hidden_dim] * num_hidden, act=act, cat_t=cat_t, ode_cfg=ode_cfg).to(device)
            ckpt_path = find_block_ckpt(master_dir, l_now, seed)
            if ckpt_path is None:
                raise FileNotFoundError(f"No checkpoint for block {l_now} in {master_dir}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if load_ema and "ema_state_dict" in ckpt:
                flow.load_state_dict(ckpt["ema_state_dict"], strict=True)
                tag = "EMA"
            else:
                flow.load_state_dict(ckpt["state_dict"], strict=True)
                tag = "raw"
            flow.eval()
            flow_ls.append(flow)
            print(f"[eval] loaded block {l_now} ({tag}) from {os.path.basename(ckpt_path)}")

        test_nll = eval_nll_loader(test_dl, flow_ls[:-1], flow_ls[-1], device=device)
        print("[final] test NLL:", test_nll)
        return

    # ---- training blocks ----
    flow_ls: List[CNF] = []
    for l in range(1, L + 2):
        
        hk = hks[l - 1]
        max_batch = int(max_batch_ls if isinstance(max_batch_ls, int) else max_batch_ls[l - 1])

        print(f"\n####### Block {l} / {L+1} with hk = {hk}")

        flow = CNF(2, [hidden_dim] * num_hidden, act=act, cat_t=cat_t, ode_cfg=ode_cfg).to(device)
        ema = deepcopy(flow).to(device).eval() if use_ema else None

        # warm start from previous trained block
        if l > 1 and warm_start and len(flow_ls) == (l - 1):
            flow.load_state_dict(flow_ls[-1].state_dict(), strict=True)
            if use_ema:
                ema.load_state_dict(flow_ls[-1].state_dict(), strict=True)

        # push train data through previous (latest) block to form current block's input
        if l > 1:
            if all_on_device:
                train_x_dev = push_tensor_by_current(flow_ls[-1], train_x_dev, batch_size=batch_push_size)
                print(f"[push] got {tuple(train_x_dev.shape)} on {train_x_dev.device}")
            else:
                push_dl = DataLoader(
                    train_dl.dataset,
                    batch_size=batch_push_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=train_dl.num_workers,
                    pin_memory=train_dl.pin_memory,
                    persistent_workers=getattr(train_dl, "persistent_workers", False),
                )
                pool_cpu = push_loader_by_current(flow_ls[-1], push_dl, device=device)
                print(f"[push] expected N={len(push_dl.dataset)} got {tuple(pool_cpu.shape)}")
                train_dl = make_tensor_loader(pool_cpu, batch_size=batch_size, shuffle=True,
                                              drop_last=True, num_workers=0, pin_memory=pin_memory)
        
        loss_fn = InterFlowLoss(flow, beta_cfg=beta_cfg, interp=interp).to(device)
        optimizer = torch.optim.Adam(flow.parameters(), lr=float(cfg["training"]["lr"]))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg["training"]["lr_step"]),
            gamma=float(cfg["training"]["lr_decay"]),
        )

        # resume
        sdict_path = os.path.join(master_dir, f"state_dict_block_{l}_seed{seed}.pt")
        resume_path = find_block_ckpt(master_dir, l, seed)

        start_batch = 0
        loss_ls = []
        if resume and resume_path is not None:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            flow.load_state_dict(ckpt["state_dict"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if use_ema:
                if "ema_state_dict" in ckpt:
                    ema.load_state_dict(ckpt["ema_state_dict"], strict=True)
                    ema_tag = "ema_state_dict"
                else:
                    ema = deepcopy(flow).to(device).eval()
                    ema_tag = "state_dict"
            else:
                ema_tag = "raw(no-ema)"

            start_batch = int(ckpt.get("batch", 0))
            loss_ls = list(ckpt.get("loss_ls", []))
            print(f"[resume] {os.path.basename(resume_path)} loaded ({ema_tag}) | start batch {start_batch}")
        elif resume:
            print(f"[resume] requested but no checkpoint found for block {l}; starting from scratch")

        if all_on_device:
            train_it = infinite_tensor_batches(train_x_dev, batch_size=batch_size, shuffle=True, drop_last=True)
            indep_it = infinite_tensor_batches(train_x_dev, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            train_it = infinite_loader(train_dl)
            indep_it = infinite_loader(train_dl)

        # plotting params
        n_plot = int(cfg.get("visualize", {}).get("n_plot", 5000))
        scat_s = float(cfg.get("visualize", {}).get("scatter_s", 0.25))
        out_block_dir = os.path.join(master_dir, f"Block_{l}")

        last_batch = start_batch - 1
        for batch in tqdm(range(start_batch, max_batch), ncols=88):
            last_batch = batch

            if all_on_device:
                x0 = next(train_it)
            else:
                x0 = unwrap_x(next(train_it)).to(device, dtype=torch.float32, non_blocking=True)  # (B,2)
            z = torch.randn_like(x0)

            # sample x1
            if l == L + 1:
                x1 = z
            else:
                B = x0.shape[0]
                if coupling == "dependent":
                    x0p = x0
                elif coupling == "shuffled":
                    x0p = x0[torch.randperm(B, device=device)]
                elif coupling == "independent":
                    if all_on_device:
                        x0p = next(indep_it)
                    else:
                        x0p = unwrap_x(next(indep_it)).to(device, dtype=torch.float32, non_blocking=True)
                else:
                    raise ValueError(f"Unknown coupling mode: {coupling}")
                x1 = OU_at_t(x0p, hk, z)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                loss = loss_fn(x0, x1)

            loss_ls.append(float(loss.item()))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            did_step = True
            total_norm = None
            if use_grad_clip:
                total_norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), grad_clip)
                if batch % 1000 == 0:
                    print(f"[grad-norm] {total_norm:.3f}")

            if total_norm is not None and (not torch.isfinite(total_norm)):
                did_step = False
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
            else:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            if did_step and use_ema:
                update_ema(ema, flow, decay=ema_decay)

            if (batch % viz_freq == 0 or batch == max_batch - 1) and batch > 0:
                is_final = (batch == max_batch - 1)
                os.makedirs(out_block_dir, exist_ok=True)

                # plot loss (overwrite) 
                plot_loss_curve(loss_ls, out_dir=out_block_dir, suffix=("final" if is_final else "latest"))

                # save ckpt
                save_path = sdict_path if batch == max_batch - 1 else sdict_path.replace(".pt", "_latest.pt")
                ckpt_out = {
                    "state_dict": flow.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "batch": last_batch + 1,
                    "loss_ls": loss_ls,
                }
                if use_ema:
                    ckpt_out["ema_state_dict"] = ema.state_dict()
                torch.save(ckpt_out, save_path)

                # quick viz       
                use_model = (ema if load_ema else flow)
                viz_block_samples(
                    xte_cpu,
                    yte_cpu=yte_cpu,
                    prev_blocks=flow_ls,
                    cur_block=use_model,
                    hk=hk,
                    device=device,
                    out_dir=out_block_dir,
                    tag=("final" if is_final else "latest"),
                    n_plot=n_plot,
                    s=scat_s,
                )

                # NLL on a subset of test
                n_eval = int(cfg.get("eval", {}).get("n_eval", min(5000, nte)))
                eval_bs = int(cfg.get("eval", {}).get("eval_batch_size", 1024))
                if all_on_device:
                    k = min(int(n_eval), int(xte_dev.shape[0]))
                    idx = torch.randperm(int(xte_dev.shape[0]), device=device)[:k]
                    x_sub = xte_dev.index_select(0, idx)
                    nll_te = eval_nll_tensor(x_sub, flow_ls, use_model, batch_size=eval_bs)
                    print(f"[NLL subset] test({k}): {nll_te:.4f}")
                else:
                    test_eval_dl = make_subset_loader(test_dl, n_eval=n_eval, batch_size=eval_bs)
                    nll_te = eval_nll_loader(test_eval_dl, flow_ls, use_model, device=device)
                    print(f"[NLL subset] test({len(test_eval_dl.dataset)}): {nll_te:.4f}")

        print(f"Done training block {l}.")
        blk = deepcopy(ema if load_ema else flow).to(device).eval()
        flow_ls.append(blk)

        
        # after final block, plot confidence regions
        if l == L + 1:
            viz_confidence_regions(
                prev_blocks=flow_ls[:-1],
                cur_block=flow_ls[-1],
                device=device,
                out_path=os.path.join(master_dir, "Conf_region.png"),
            )
        
            
    print("\nAll blocks finished. Final test NLL...")
    if all_on_device:
        test_nll = eval_nll_tensor(xte_dev, flow_ls[:-1], flow_ls[-1], batch_size=test_batch_size)
    else:
        test_nll = eval_nll_loader(test_dl, flow_ls[:-1], flow_ls[-1], device=device)
    print("[final] test NLL:", test_nll)
    
    # write final NLL to text file
    with open(os.path.join(master_dir, f"test_nll_seed{seed}.txt"), "w") as ftxt:
        ftxt.write(f"test nll: {test_nll}\n")
    
    # final generated samples (scatter) in master_dir    
    n_plot_final = int(cfg.get("visualize", {}).get("n_plot", 5000))
    pts = sample_final(n_plot_final,flow_ls=flow_ls,device=device)
    scatter_save(os.path.join(master_dir, "Final_Generated.png"), pts, "Final generated", s=0.25)

if __name__ == "__main__":
    main()
