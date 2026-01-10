import math
import os
import re
import argparse
import yaml
from copy import deepcopy
from collections import OrderedDict
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import torchdiffeq as tdeq
import matplotlib.pyplot as plt

import datasets_tabular  

# -------------------------
# utils 
# -------------------------

def unwrap_x(batch):
    # TensorDataset yields (x,), legacy may yield x
    if isinstance(batch, (list, tuple)) and len(batch) >= 1:
        return batch[0]
    return batch

def infinite_loader(dl):
    while True:
        for b in dl:
            yield b

def make_tensor_loader(x_cpu: torch.Tensor, batch_size: int, *,
                       shuffle: bool = True, drop_last: bool = True,
                       num_workers: int = 0, pin_memory: bool = True):
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
    # Fallback: materialize by iterating once (should be rare)
    xs = []
    for b in DataLoader(ds, batch_size=65536, shuffle=False, drop_last=False, num_workers=0):
        xs.append(unwrap_x(b))
    return torch.cat(xs, dim=0)

def infinite_tensor_batches(x_dev: torch.Tensor, batch_size: int, *,
                           shuffle: bool = True, drop_last: bool = True):
    """Yield GPU batches by indexing a GPU resident tensor."""
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
      

# -------------------------
# flow network, fully connected MLP
# -------------------------

def count_params(model):
    tot_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot_params/1000

def build_mlp(in_features: int, hidden_features: List[int], *, cat_t: str = "first"):
    '''
    cat_t:
      - "first": concatenate time only to the first layer input
      - "all":   concatenate time to every layer input
    '''
    if cat_t not in ("first", "all"):
        raise ValueError(f"cat_t must be 'first' or 'all', got {cat_t}")
    layers = nn.ModuleList()
    for i, (a, b) in enumerate(zip([in_features] + hidden_features, hidden_features + [in_features])):
        add_t = (cat_t == "all") or (cat_t == "first" and i == 0)
        layers.append(nn.Linear(a + (1 if add_t else 0), b))
    return layers

class CNF(nn.Module):
    def __init__(self, data_dim: int, hidden_features: List[int], *,
                 act: str = "relu", cat_t: str = "first", ode_cfg: Optional[dict] = None):
        super().__init__()
        self.data_dim = int(data_dim)
        self.net = build_mlp(self.data_dim, hidden_features, cat_t=cat_t)
        self.cat_t = cat_t
        self.act = nn.ReLU() if act.lower() == "relu" else nn.ELU()

        self.ode_cfg = ode_cfg or {"method": "dopri5", "tol": 1e-5, "num_steps": 100}

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        # t: (B,1) during training, or scalar during odeint
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
        '''
        NOTE: This is O(D) backprops per batch and can be heavy when D is large.
        '''
        self.eval()
        device = x.device
        D = x.shape[-1]

        # shape: (D, B, D) for is_grads_batched=True
        I = torch.eye(D, device=device, dtype=x.dtype)
        I = I.expand(x.shape + (D,)).movedim(-1, 0)

        def augmented(t, x_ladj):
            xt, ladj = x_ladj
            with torch.enable_grad():
                xt = xt.requires_grad_()
                dx = self(t, xt)
            jacobian = torch.autograd.grad(dx, xt, I, is_grads_batched=True, create_graph=True)[0]
            trace = torch.einsum("i...i", jacobian)
            return dx, trace

        ladj0 = torch.zeros(x.shape[0], device=device, dtype=x.dtype)
        tspan = torch.linspace(0.0, 1.0, 150, device=device, dtype=x.dtype)
        with torch.no_grad():
            z, ladj = tdeq.odeint(augmented, (x, ladj0), tspan, method="euler")
        z = z[-1]
        ladj = ladj[-1]

        if not return_full:
            return ladj

        log_qz = Normal(0.0, z.new_tensor(1.0)).log_prob(z).sum(dim=-1)
        return log_qz + ladj

# -------------------------
# loss 
# -------------------------

class InterFlowLoss(nn.Module):
    """
    Flow Matching loss with selectable interpolation:
      - interp="linear": I_t = x0 + t (x1-x0), dI_t = x1-x0
      - interp="trig":   I_t = cos(pi t/2)x0 + sin(pi t/2)x1,
                         dI_t = (pi/2)(cos(pi t/2)x1 - sin(pi t/2)x0)
    """

    def __init__(
        self,
        v: nn.Module,
        beta_cfg: Optional[dict] = None,
        interp: str = "linear",
        num_tk: int = 1,          # <-- optional: average over K t-samples (vectorized)
    ):
        super().__init__()
        self.v = v

        # t sampling config
        self.beta_cfg = beta_cfg
        if beta_cfg is not None and beta_cfg.get("enabled", False):
            self.register_buffer("_beta_alpha", torch.tensor(float(beta_cfg["alpha"])))
            self.register_buffer("_beta_beta",  torch.tensor(float(beta_cfg["beta"])))
        else:
            self._beta_alpha = None
            self._beta_beta = None

        # interpolation
        interp = interp.lower()
        if interp not in ("linear", "trig"):
            raise ValueError(f"Unknown interp='{interp}'. Use 'linear' or 'trig'.")
        self.interp = interp

        # number of time samples per pair
        self.num_tk = int(num_tk)
        if self.num_tk < 1:
            raise ValueError("num_tk must be >= 1")

    def sample_t(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return t of shape (B,1) on (device,dtype)."""
        if self._beta_alpha is not None:
            alpha = self._beta_alpha.to(device=device, dtype=dtype)
            beta  = self._beta_beta.to(device=device, dtype=dtype)
            return torch.distributions.Beta(alpha, beta).sample((B, 1))
        return torch.rand(B, 1, device=device, dtype=dtype)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        x0, x1: (B, D) or (B, C, H, W) etc. Works for any shape with batch dim first.
        """
        B = x0.shape[0]
        device, dtype = x0.device, x0.dtype

        K = self.num_tk
        if K == 1:
            t = self.sample_t(B, device, dtype)  # (B,1)
            t_view = t.view(B, *([1] * (x0.ndim - 1)))  # broadcast to x0 shape

            if self.interp == "linear":
                dIt = x1 - x0
                It = x0 + t_view * dIt
            else:
                c = torch.cos((math.pi / 2.0) * t).view(B, *([1] * (x0.ndim - 1)))
                s = torch.sin((math.pi / 2.0) * t).view(B, *([1] * (x0.ndim - 1)))
                It = c * x0 + s * x1
                dIt = (math.pi / 2.0) * (c * x1 - s * x0)

            vout = self.v(t, It)  # expects t: (B,1), It: (B,...)
            return (vout - dIt).pow(2).mean()

        # -------- vectorized multi-t (no Python loop) --------
        # sample t: (K,B,1)
        t = self.sample_t(B, device, dtype).unsqueeze(0).expand(K, B, 1).contiguous()
        # reshape to (K*B,1) for a single model call
        t_flat = t.reshape(K * B, 1)

        # repeat x0/x1 to (K*B, ...)
        x0_rep = x0.unsqueeze(0).expand(K, *x0.shape).reshape(K * B, *x0.shape[1:])
        x1_rep = x1.unsqueeze(0).expand(K, *x1.shape).reshape(K * B, *x1.shape[1:])

        # broadcast view for multiplying into x tensors
        t_view = t_flat.view(K * B, *([1] * (x0.ndim - 1)))

        if self.interp == "linear":
            dIt = x1_rep - x0_rep
            It = x0_rep + t_view * dIt
        else:
            c = torch.cos((math.pi / 2.0) * t_flat).view(K * B, *([1] * (x0.ndim - 1)))
            s = torch.sin((math.pi / 2.0) * t_flat).view(K * B, *([1] * (x0.ndim - 1)))
            It = c * x0_rep + s * x1_rep
            dIt = (math.pi / 2.0) * (c * x1_rep - s * x0_rep)

        vout = self.v(t_flat, It)
        return (vout - dIt).pow(2).mean()

# -------------------------
# push + eval (NLL)
# -------------------------

@torch.no_grad()
def push_loader_by_current(flow: CNF, dl: DataLoader, *, device: torch.device):
    flow.eval()
    outs = []
    for batch in tqdm(dl, desc="Pushforward all", ncols=88):
        x = unwrap_x(batch).to(device, dtype=torch.float32, non_blocking=True)
        traj = flow.front_or_back(x, towards_one=True)
        y = traj[-1].detach().cpu()
        outs.append(y)
    return torch.cat(outs, dim=0)

@torch.no_grad()
def eval_nll_loader(test_dl, prev_blocks: List[CNF], cur_block: CNF, *,
                    device: torch.device) -> float:
    """
    Loader version of eval_nll:
      - iterates over batches from test_dl (CPU -> GPU per batch)
      - computes NLL for composed flow (prev_blocks then cur_block)
      - returns sample-weighted mean NLL over the whole loader
    """
    cur_block.eval()
    for b in prev_blocks:
        b.eval()

    total_nll = 0.0
    total_n = 0

    for (x_cpu,) in tqdm(test_dl, desc="Evaluating NLL", leave=False):   # TensorDataset yields 1-tuples
        bs = x_cpu.shape[0]
        x = x_cpu.to(device, dtype=torch.float32, non_blocking=True)

        log_p_prev = 0.0
        for blk in prev_blocks:
            log_p_prev += blk.log_prob(x, return_full=False).mean().item()
            x = blk.front_or_back(x, towards_one=True)[-1]

        lp = cur_block.log_prob(x, return_full=True).mean().item()
        lp = lp + log_p_prev

        nll_batch = -lp
        total_nll += nll_batch * bs
        total_n += bs

    return float(total_nll / max(total_n, 1))


@torch.no_grad()
def push_tensor_by_current(flow: CNF, x_dev: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    """Pushforward a whole GPU tensor through `flow` in chunks; returns GPU tensor."""
    flow.eval()
    outs = []
    n = x_dev.shape[0]
    for i in tqdm(range(0, n, batch_size), desc="Pushforward all (GPU)", ncols=88):
        xb = x_dev[i:i+batch_size]
        y = flow.front_or_back(xb, towards_one=True)[-1]
        outs.append(y.detach())
    return torch.cat(outs, dim=0)

@torch.no_grad()
def eval_nll_tensor(x_dev: torch.Tensor, prev_blocks: List[CNF], cur_block: CNF, *,
                    batch_size: int) -> float:
    """Evaluate NLL over a GPU tensor, batching internally."""
    cur_block.eval()
    for b in prev_blocks:
        b.eval()

    total_nll = 0.0
    total_n = 0
    n = x_dev.shape[0]
    for i in tqdm(range(0, n, batch_size), desc="Evaluating NLL (GPU)", leave=False):
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

def make_subset_loader(dl: DataLoader, n_eval: int,
                       batch_size: Optional[int] = None,
                       num_workers: Optional[int] = None,
                       pin_memory: Optional[bool] = None) -> DataLoader:    
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
# other training related
# -------------------------

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

def make_warmup_lr(warmup_steps: int):
    def warmup_lr(step: int):
        if warmup_steps <= 0:
            return 1.0
        return min(step, warmup_steps) / float(warmup_steps)
    return warmup_lr

def find_block_ckpt(master_dir: str, block_id: int, seed: int):
    '''
    Priority:
      1) state_dict_block_{block_id}_seed{seed}.pt
      2) state_dict_block_{block_id}_seed{seed}_latest.pt
      3) state_dict_block_{block_id}_seed{seed}_*.pt  (largest trailing number)
    '''
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
# main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Tabular LFM training")
    parser.add_argument("--hyper_param_config", type=str, default="config/tabular_lfm.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--all_on_device", action="store_true", help="Keep train/val/test tensors on device and batch by GPU indexing (faster if data fits GPU)")
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

    base_dir = cfg.get("base_dir", "results_tabular")
    save_dir = cfg.get("save_dir", "debug_run")
    master_dir = os.path.join(base_dir, save_dir)
    os.makedirs(master_dir, exist_ok=True)
    with open(os.path.join(master_dir, "config.txt"), "w") as f:
        f.write(yaml.dump(cfg, default_flow_style=False))

    # ---- data via datasets_tabular (CPU dataset, batch -> GPU) ----
    dataname = cfg["data"]["dataname"]
    data_root = cfg["data"].get("data_root", "./data")

    # loaders
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["data"].get("num_workers", 2))
    pin_memory = bool(cfg["data"].get("pin_memory", True))
    test_batch_size = int(cfg.get("eval", {}).get("test_batch_size", batch_size))

    train_dl, val_dl, test_dl, info = datasets_tabular.make_tabular_loaders(
        dataname,
        data_root=data_root,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_dl_raw = train_dl #save a copy of dl for the original data
    
    all_on_device = bool(args.all_on_device or cfg.get("data", {}).get("all_on_device", False))
        # if true, keep whole splits on device, other wise DataLoader (CPU->GPU each batch)
    
    if all_on_device:
        # Materialize CPU tensors from TensorDataset and move once
        train_x_raw = get_ds_tensor(train_dl_raw).to(device=device, dtype=torch.float32)
        val_x_dev   = get_ds_tensor(val_dl).to(device=device, dtype=torch.float32)
        test_x_dev  = get_ds_tensor(test_dl).to(device=device, dtype=torch.float32)
        # Current train pool for this block (will be updated by pushforward across blocks)
        train_x_dev = train_x_raw
        print(f"[all_on_device] train={tuple(train_x_dev.shape)} val={tuple(val_x_dev.shape)} test={tuple(test_x_dev.shape)} on {device}")
    else:
        train_x_dev = None
        val_x_dev = None
        test_x_dev = None
    
    data_dim = int(info["n_dims"])
    print(f"[data] dim={data_dim} train={info['n_train']} test={info['n_test']} (val={info['n_val']})")


    # training config
    L = int(cfg["model"]["L"])  # blocks are 1..L+1
    hks = cfg["model"]["hks"]
    assert len(hks) == L + 1, "Expected hks length L+1 (one per block)."

    max_batch_ls = cfg["training"]["max_batch"]
    #lr = float(cfg["training"]["lr"])
    warmup_steps = int(cfg["training"].get("warmup", 0))
    
    resume = bool(cfg["training"].get("resume", False))
    eval_only = bool(cfg["training"].get("eval_only", False))
    warm_start = bool(cfg["training"].get("warm_start", False))

    use_grad_clip = bool(cfg["training"].get("use_grad_clip", True))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    use_ema = bool(cfg["training"].get("use_ema", True))
    load_ema = bool(cfg["training"].get("load_ema", True)) and use_ema
    ema_decay = float(cfg["training"].get("ema_decay", 0.9999)) if use_ema else 0.0

    batch_push_size = int(cfg["training"].get("batch_push_size", 8192))
    viz_freq = int(cfg.get("visualize", {}).get("viz_freq", 2000))

    coupling = str(cfg["model"].get("coupling", "dependent")).lower()
    interp = str(cfg["model"].get("interp", "linear")).lower()

    num_tk   = int(cfg["model"].get("num_tk", 1))
    
    beta_cfg = None
    if cfg["model"].get("beta_sample_t", False):
        beta_cfg = {"enabled": True, "alpha": cfg["model"]["alpha"], "beta": cfg["model"]["beta"]}
        print(f"[t-sampling] Beta({beta_cfg['alpha']},{beta_cfg['beta']}) enabled")

    # model config
    hidden_dim = int(cfg["model"]["hidden_dim"])
    num_hidden = int(cfg["model"]["num_hidden"])
    act = str(cfg["model"].get("act", "relu"))
    cat_t = str(cfg["model"].get("cat_t", "first"))  # "first" or "all"
    ode_cfg = cfg.get("ode", {"method": "dopri5", "tol": 1e-5, "num_steps": 100})

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ---- eval-only path ----
    if eval_only:
        flow_ls: List[CNF] = []
        for l_now in range(1, L + 2):
            flow = CNF(data_dim, [hidden_dim] * num_hidden, act=act, cat_t=cat_t, ode_cfg=ode_cfg).to(device)
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

        if all_on_device:
            test_nll = eval_nll_tensor(test_x_dev, flow_ls[:-1], flow_ls[-1], batch_size=test_batch_size)
        else:
            test_nll = eval_nll_loader(test_dl, flow_ls[:-1], flow_ls[-1], device=device)
        print("[final] test NLL:", test_nll)

        fname = f"test_nll_seed{seed}.txt"
        with open(os.path.join(master_dir, fname), "w") as ftxt:
            ftxt.write(f"test nll: {test_nll}\n")
        return

    # ---- training ----
    flow_ls: List[CNF] = []
    for l in range(1, L + 2):
        
        hk = hks[l - 1]
        max_batch = int(max_batch_ls if isinstance(max_batch_ls, int) else max_batch_ls[l - 1])
        
        print(f"\n####### Block {l} / {L+1} with hk = {hk}")

        flow = CNF(data_dim, [hidden_dim] * num_hidden, act=act, cat_t=cat_t, ode_cfg=ode_cfg).to(device)
        print(f'Number of parameters: {count_params(flow):.3f}k')
        print(flow)
        
        ema = deepcopy(flow).to(device).eval() if use_ema else None


        if l > 1 and warm_start and len(flow_ls) == (l - 1):
            flow.load_state_dict(flow_ls[-1].state_dict(), strict=True)
            if use_ema:
                ema.load_state_dict(flow_ls[-1].state_dict(), strict=True)

        # pushing for the next block
        if l > 1:
            assert len(flow_ls) == (l - 1)
            if all_on_device:
                # push the *GPU resident* training pool through previous block, keep result on GPU
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
        loss_fn = InterFlowLoss(flow, beta_cfg=beta_cfg, interp=interp, num_tk=num_tk).to(device)

        optimizer = torch.optim.Adam(flow.parameters(), lr=float(cfg["training"]["lr"]))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=cfg['training']['lr_step'], 
                                                    gamma=cfg['training']['lr_decay'])
        

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

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        last_batch = start_batch - 1
        for batch in tqdm(range(start_batch, max_batch), ncols=88):
            last_batch = batch
            if all_on_device:
                x0 = next(train_it)  # already on device, float32
            else:
                x0 = unwrap_x(next(train_it)).to(device, dtype=torch.float32, non_blocking=True)  # (B,D)

            z = torch.randn_like(x0)
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

                shrink = math.exp(-hk)
                sigma = math.sqrt(max(0.0, 1.0 - shrink ** 2))
                x1 = shrink * x0p + sigma * z

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
                decay = 0.0 if batch < warmup_steps else ema_decay
                update_ema(ema, flow, decay=decay)

            if (batch % viz_freq == 0 or batch == max_batch - 1) and batch > 0:
                if device.type == "cuda":
                    print(f"[GPU mem] allocated={torch.cuda.memory_allocated()/1024**2:.1f} MB | "
                          f"reserved={torch.cuda.memory_reserved()/1024**2:.1f} MB")
                    
                # plot loss
                plt.figure()
                plt.plot(loss_ls)
                plt.xlabel("Batch")
                plt.ylabel("Loss")
                plt.tight_layout()
                plt.savefig(os.path.join(master_dir, f"Block_{l}_Losses_loss_latest.png"))
                plt.close()

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

                # evaluate nll on train and val sets
                n_eval = int(cfg.get("eval", {}).get("n_eval", 5000))
                eval_bs = int(cfg.get("eval", {}).get("eval_batch_size", 1024))

                if all_on_device:
                    # random subsets on GPU (no CPU DataLoader)
                    n_tr = train_x_raw.shape[0]
                    n_va = val_x_dev.shape[0]
                    k_tr = min(int(n_eval), int(n_tr))
                    k_va = min(int(n_eval), int(n_va))
                    idx_tr = torch.randperm(n_tr, device=device)[:k_tr]
                    idx_va = torch.randperm(n_va, device=device)[:k_va]
                    x_tr = train_x_raw.index_select(0, idx_tr)
                    x_va = val_x_dev.index_select(0, idx_va)

                    nll_tr = eval_nll_tensor(x_tr, flow_ls, ema if load_ema else flow, batch_size=eval_bs)
                    nll_va = eval_nll_tensor(x_va, flow_ls, ema if load_ema else flow, batch_size=eval_bs)
                    print(f"[NLL subset] train({k_tr}): {nll_tr:.4f}  val({k_va}): {nll_va:.4f}")
                else:
                    train_eval_dl = make_subset_loader(train_dl_raw, n_eval=n_eval, batch_size=eval_bs)
                    val_eval_dl   = make_subset_loader(val_dl,   n_eval=n_eval, batch_size=eval_bs)
                    nll_tr = eval_nll_loader(train_eval_dl, flow_ls, ema if load_ema else flow, device=device)
                    nll_va = eval_nll_loader(val_eval_dl,   flow_ls, ema if load_ema else flow, device=device)
                    print(f"[NLL subset] train({len(train_eval_dl.dataset)}): {nll_tr:.4f}  "
                          f"val({len(val_eval_dl.dataset)}): {nll_va:.4f}")

        print(f"Done training block {l}.")
        blk = deepcopy(ema if load_ema else flow).to(device).eval()
        flow_ls.append(blk)

        if device.type == "cuda":
            print(f"[block {l} GPU peak] {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")

    print("\nAll blocks finished. Final test NLL...")
    if all_on_device:
        test_nll = eval_nll_tensor(test_x_dev, flow_ls[:-1], flow_ls[-1], batch_size=test_batch_size)
    else:
        test_nll = eval_nll_loader(test_dl, flow_ls[:-1], flow_ls[-1], device=device)
    print("[final] test NLL:", test_nll)

    fname = f"test_nll_seed{seed}.txt"
    with open(os.path.join(master_dir, fname), "w") as ftxt:
        ftxt.write(f"test nll: {test_nll}\n")
    

if __name__ == "__main__":
    main()