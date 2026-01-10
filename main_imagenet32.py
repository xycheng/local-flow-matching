import math
import matplotlib.pyplot as plt
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

from torch.amp import autocast, GradScaler

from tqdm import tqdm
from typing import *
import torchdiffeq as tdeq
import os
import data

from copy import deepcopy
from collections import OrderedDict
from torchvision.utils import make_grid, save_image


# data
from torch.utils.data import TensorDataset, DataLoader
from datasets_images import make_imagenet32_loaders, infinite_loader

# Unet
from unet.unet import UNetModelWrapper

# fid
from cleanfid import fid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# dataset
# -------------------------

# torchvision datasets yield (x, y) but your pushed pool will yield x only
def unwrap_x(batch):
    # batch may be (x, y) or just x
    if isinstance(batch, (list, tuple)) and len(batch) >= 1:
        return batch[0]
    return batch
    
def make_pool_loader(x_cpu: torch.Tensor, batch_size: int, *, shuffle=True,
                     num_workers=0, pin_memory=True, drop_last=True):
    # x_cpu: CPU tensor (N,C,H,W) in [-1,1]
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

# -------------------------
# model
# -------------------------

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

class CNF(nn.Module):
    def __init__(self, in_shape,  dim,  dim_mults, attention_resolutions,
                                 num_res_blocks, dropout, ode_cfg):
        super().__init__()
        self.net = UNetModelWrapper(
                        dim= in_shape, #input dimension (C, W , H)
                        num_res_blocks = num_res_blocks,
                        num_channels = dim, #model channels
                        channel_mult= dim_mults,
                        num_heads= 4,
                        num_head_channels= 64,
                        attention_resolutions=attention_resolutions, #string
                        dropout=dropout) 
        self.ode_cfg = ode_cfg

    def forward(self, t, x):
        # UNet expects (x, time) where time is (batch,) or scalar
        if len(t.size()) == 2:
            # t is (batch, 1), squeeze to (batch,)
            time = t.squeeze(-1)
        else:
            # t is scalar, expand to (batch,)
            batch_size = x.shape[0]
            time = torch.ones(batch_size, device=x.device) * t
        # UNet forward: (x, time, x_self_cond=None)
        return self.net(x, time)

    def front_or_back(self, z, towards_one = False):
        self.net.eval()      
        device = z.device
        first, second = (0.0, 1.0) if towards_one else (1.0, 0.0)
        
        method = self.ode_cfg["method"].lower()
        if method == "dopri5":
            rtol = float(self.ode_cfg.get("tol", 1e-5))
            atol = float(self.ode_cfg.get("tol", 1e-5))       
            tspan = torch.tensor([first, second], device=device, dtype=torch.float32)    
            with torch.no_grad():
                z_t = tdeq.odeint(self, z, tspan, method=method, 
                                   rtol =rtol, atol = atol)
            return z_t #[-1]   # final state only
            
        # ---------- fixed-step solvers ----------
        elif method in ("rk4", "euler"):
            num_steps = int(self.ode_cfg["num_steps"])
            tspan=torch.linspace(first, second, num_steps, device=device, dtype=torch.float32)
            with torch.no_grad():
                z_t = tdeq.odeint(self, z, tspan, method=method)
            return z_t
        else:
            raise ValueError(f"Unknown ODE method: {method}")    

import math
class InterFlowLoss(nn.Module):
    """
    Flow Matching loss with selectable interpolation:
      - interp="linear":      I_t = x0 + t (x1-x0),        dI_t = x1-x0
      - interp="trig":        I_t = cos(pi t/2)x0 + sin(pi t/2)x1,
                             dI_t = (pi/2)(cos(pi t/2)x1 - sin(pi t/2)x0)

    t is always sampled in [0,1]:  
      - uniform t
      - beta sampling t (per-batch per-sample)
    """
    def __init__(self, v: nn.Module, beta_cfg: Optional[dict] = None, interp: str = "linear"):
        super().__init__()
        self.v = v
        
        # sample_t
        self.beta_cfg = beta_cfg
        # buffers so they move with .to(device) / .cuda()
        if beta_cfg is not None and beta_cfg.get("enabled", False):
            self.register_buffer("_beta_alpha", torch.tensor(float(beta_cfg["alpha"])))
            self.register_buffer("_beta_beta",  torch.tensor(float(beta_cfg["beta"])))
        else:
            self._beta_alpha = None
            self._beta_beta = None

        # interpolation function
        interp = interp.lower()
        if interp not in ("linear", "trig"):
            raise ValueError(f"Unknown interp='{interp}'. Use 'linear' or 'trig'.")
        self.interp = interp

    def sample_t(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._beta_alpha is not None:
            alpha = self._beta_alpha.to(device=device, dtype=dtype)
            beta  = self._beta_beta.to(device=device, dtype=dtype)
            return torch.distributions.Beta(alpha, beta).sample((B, 1))
        else:
            return torch.rand(B, 1, device=device, dtype=dtype)
        

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        t = self.sample_t(B, x0.device, x0.dtype)          # (B,1)
        t_img = t.view(B, 1, 1, 1)                         # broadcast over image

        if self.interp == "linear":
            dIt = x1 - x0
            It  = x0 + t_img * dIt
        else:
            # trig interpolation
            # I_t = cos(pi t/2) x0 + sin(pi t/2) x1
            # dI_t = (pi/2)(cos(pi t/2) x1 - sin(pi t/2) x0)
            c = torch.cos((math.pi / 2.0) * t_img)
            s = torch.sin((math.pi / 2.0) * t_img)
            It  = c * x0 + s * x1
            dIt = (math.pi / 2.0) * (c * x1 - s * x0)

        vout = self.v(t, It)  # assumes v takes t as (B,1) and x as (B,C,H,W)
        return (vout - dIt).pow(2).mean()


class LFM(nn.Module):
    def __init__(self, flow_ls: list):
        super().__init__()
        self.flow_ls = flow_ls

    @torch.no_grad()
    def sample(self, z, return_list: bool = False, n_frames: int = 12):
        """
        If return_list=False:
            returns final sample float in [0,1] (quantized like before).
        If return_list=True:
            returns list of length n_frames spanning noise->final.
            frames are float in [0,1] by default
        """
        cur = z

        if not return_list:
            for flow in reversed(self.flow_ls):
                ret = flow.front_or_back(cur, towards_one=False)
                cur = ret[-1]
            cur = torch.clamp(cur * 0.5 + 0.5, 0.0, 1.0)
            return cur

        # Collect full trajectory across all blocks. This uses each CNF's own ode_cfg.
        traj = [cur]  # pure noise start frame
        for flow in reversed(self.flow_ls):
            ret = flow.front_or_back(cur, towards_one=False)
            traj.extend(list(ret[1:]))
            #traj.extend( ret[1:] if len(ret) > 1 else ret)
            cur = ret[-1]
        idx = _pick_evenly_spaced_indices(len(traj), n_frames)
        print("T =", len(traj), "n_frames =", n_frames)
        print("idx =", idx)            
        frames = [traj[i] for i in idx]
        frames01 = [torch.clamp(x * 0.5 + 0.5, 0.0, 1.0) for x in frames] # to [0,1]
        return frames01

def count_params(model):
    tot_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot_params/1000

@torch.no_grad()
def push_loader_by_current_prealloc(flow, dl, *, device, out_dtype=torch.float16):
    flow.eval()
    N = len(dl.dataset)
    # assume fixed shape from first batch
    first = next(iter(dl))
    x0 = unwrap_x(first)
    C, H, W = x0.shape[1:]
    pool = torch.empty((N, C, H, W), dtype=out_dtype, device="cpu")

    idx = 0
    for batch in tqdm(dl, desc="Pushforward all", ncols=88):
        x = unwrap_x(batch).to(device, dtype=torch.float32, non_blocking=True)
        traj = flow.front_or_back(x, towards_one=True)
        y = traj[-1].detach().to("cpu", dtype=out_dtype)
        bsz = y.shape[0]
        pool[idx:idx+bsz].copy_(y)
        idx += bsz
        # help GC / free references
        del traj, x, y
    assert idx == N, (idx, N)
    return pool
    

### --- check points ---
    
import re
def find_block_ckpt(master_dir, block_id, seed):
    """
    Priority:
    1) state_dict_block_{block_id}_seed{seed}.pt
    2) state_dict_block_{block_id}_seed{seed}_latest.pt
    3) state_dict_block_{block_id}_seed{seed}_*.pt  (largest trailing number)
    """
    # 1) canonical
    canonical = os.path.join(master_dir, f"state_dict_block_{block_id}_seed{seed}.pt")
    if os.path.exists(canonical):
        return canonical

    # 2) latest (rolling)
    latest = os.path.join(master_dir, f"state_dict_block_{block_id}_seed{seed}_latest.pt")
    if os.path.exists(latest):
        return latest

    # 3) legacy numbered: ..._10000.pt
    pattern = re.compile(rf"state_dict_block_{block_id}_seed{seed}_(\d+)\.pt")
    best_fname = None
    best_step = -1
    for fname in os.listdir(master_dir):
        m = pattern.fullmatch(fname)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best_fname = fname
    if best_fname is None:
        return None

    return os.path.join(master_dir, best_fname)

# -------------------------
# sampling and fid 
# -------------------------

# for panel plot
def _pick_evenly_spaced_indices(T: int, n_frames: int):
    """
    Pick n_frames indices from [0, T-1], always including endpoints.
    Safe for any T>=1.
    """
    if n_frames <= 1:
        return [0]
    if T <= 1:
        return [0] * n_frames
    idx = np.linspace(0, T - 1, n_frames)
    idx = np.round(idx).astype(int)
    idx[0] = 0
    idx[-1] = T - 1
    return idx.tolist()
    

@torch.no_grad()
def plot_image_chain(lfm: LFM, in_shape, num_images: int = 5, 
                     num_grid_points: Optional[int] = 5, master_dir: Optional[str] = None):
    """
    Plots a grid of trajectories (rows = images, cols = frames) from pure noise -> final sample.
    """
    # Infer device from model (avoid relying on a global `device`)
    try:
        device = next(lfm.parameters()).device
    except StopIteration:
        # If lfm has no parameters for some reason, fall back to cuda if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    C, H, W = in_shape
    z = torch.randn(num_images, C, H, W, device=device)
    lfm.eval()
    flow_ls = list(getattr(lfm, "flow_ls", []))
    if not flow_ls:
        raise ValueError("lfm.flow_ls is empty.")

    # grid per block
    grid = torch.linspace(0.0,1.0,steps=num_grid_points, device=device, dtype=torch.float32) # will has 0, 1 as endpoints
    tspan_block = torch.flip(grid, dims=[0])  # from 1 to 0

    # Collect frames across blocks
    frames = []
    cur = z
    frames.append(cur)  # start (noise)
    # sampling path uses reversed blocks
    for bi, flow in enumerate(reversed(flow_ls)):
        flow.net.eval()
        ode_cfg = getattr(flow, "ode_cfg", {})
        method = str(ode_cfg.get("method", "dopri5")).lower()
        if method == "dopri5":
            tol = float(ode_cfg.get("tol", 1e-5))
            rtol = tol
            atol = tol
            z_t = tdeq.odeint(flow, cur, tspan_block, method="dopri5", rtol=rtol, atol=atol)
        elif method in ("rk4", "euler"):
            # for fixed-step methods, this also returns at the requested grid
            z_t = tdeq.odeint(flow, cur, tspan_block, method=method)
        else:
            raise ValueError(f"Unknown ODE method in ode_cfg: {method}")

        # z_t is (T, B, C, H, W). Avoid duplicating block boundary:
        # first element equals cur, so skip it.
        frames.extend([z_t[k] for k in range(1, z_t.shape[0])])
        cur = z_t[-1]
        
    # Convert to [0,1] for plotting
    frames01 = [torch.clamp(x * 0.5 + 0.5, 0.0, 1.0) for x in frames]
    T = len(frames01)
    B = num_images
    fig, axes = plt.subplots(nrows=B,ncols=T,figsize=(1.6 * T, 1.6 * B),squeeze=False)
    for b in range(B):
        for t in range(T):
            ax = axes[b][t]
            xt = frames01[t][b].detach().cpu()
            if xt.shape[0] == 1:
                ax.imshow(xt[0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(xt.permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if b == 0:
                ax.set_title(f"{t}", fontsize=8)
        axes[b][0].set_ylabel(f"z#{b}", rotation=0, labelpad=20, va="center", fontsize=9)
    plt.tight_layout()
    if master_dir:
        outpath = os.path.join(master_dir, "image_chain.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return outpath
    plt.close(fig)
    return None

def make_fid_gen(lfm, in_shape, device):
    """
    Returns a CleanFID-compatible generator:
    gen(z) -> torch.uint8 images in (B,3,32,32), range [0,255]
    """
    C, H, W = in_shape
    @torch.no_grad()
    def gen(z):
        # z is provided by cleanfid; we only use it to infer batch size
        B = z.shape[0] if torch.is_tensor(z) else len(z)
        z_img = torch.randn(B, C, H, W, device=device)
        img = lfm.sample(z_img, return_list=False)  # (B,3,32,32) float in [0,1]
        # Ensure NCHW
        assert img.ndim == 4 and img.shape[1] == 3, f"Expected (B,3,H,W), got {img.shape}"
        # convert to uint8 [0,255]
        x_u8 = (img * 255.0).round().clamp(0, 255).to(torch.uint8)
        return x_u8
    return gen

def run_final_eval(*, root: str, master_dir: str, seed: int, L: int,
                   in_shape, hidden_dim: int, dim_mults, attention_resolutions,
                   num_res_blocks: int, dropout: float, ode_cfg: dict,
                   load_ema: bool, cal_fid: bool, fid_num_samples: int, gen_bs: int, device):
    print("[final-eval] Loading trained blocks...")

    flow_ls = []
    for l_now in range(1, L + 2):  # blocks are 1..L+1
        f = CNF(in_shape, hidden_dim, dim_mults, attention_resolutions,
                num_res_blocks, dropout, ode_cfg).to(device)
        
        ckpt_path = find_block_ckpt(master_dir, l_now, seed)
        if ckpt_path is None:
            raise FileNotFoundError(f"[final-eval] No checkpoint found for block {l_now}, seed {seed}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        if load_ema and ("ema_state_dict" in ckpt):
            f.load_state_dict(ckpt["ema_state_dict"], strict=True)
            tag = "EMA"
        else:
            f.load_state_dict(ckpt["state_dict"], strict=True)
            tag = "raw"

        f.eval()
        flow_ls.append(f)
        print(f"[final-eval] loaded block {l_now} ({tag})")

    lfm = LFM(flow_ls).to(device)
    lfm.eval()

    # ---- grid ----
    nrow = 6
    num_gen = nrow * nrow
    z = torch.randn(num_gen, in_shape[0], in_shape[1], in_shape[2], device=device)
    x01 = lfm.sample(z, return_list=False) # in [0,1]
    grid = make_grid(x01, nrow=nrow)
    out_grid = os.path.join(master_dir, f"grid_{nrow}x{nrow}_seed{seed}.png")
    save_image(grid, out_grid)
    print(f"[final-eval] saved grid -> {out_grid}")

    # ---- panel/chain ----
    out_chain = plot_image_chain(lfm, in_shape, num_images=5, num_grid_points=5, master_dir=master_dir)
    print(f"[final-eval] saved chain -> {out_chain}")

    # ---- FID ----
    if cal_fid:
        gen = make_fid_gen(lfm, in_shape=in_shape, device=device)

        stats_name = "imagenet32_val_cached_stats"   
        real_dir   = os.path.join(root, "imagenet32_val_ref_png")  

        # import here to avoid threading train_ds through everything
        from datasets_images import ImageNet32Binary, dump_real_png_subset # datasets_images.py
        
        if not fid.test_stats_exists(stats_name, mode="legacy_tensorflow"):
            if not os.path.isdir(real_dir) or len(os.listdir(real_dir)) == 0:
                val_ds = ImageNet32Binary(root=root, split="val", transform=None)
                dump_real_png_subset(val_ds, real_dir, n=50000)  # val has exactly 50k
            fid.make_custom_stats(stats_name, real_dir, mode="legacy_tensorflow")
        
        score = fid.compute_fid(
            gen=gen,
            dataset_name=stats_name,
            dataset_split="custom",
            num_gen=fid_num_samples,
            batch_size=gen_bs,
            mode="legacy_tensorflow",
        )
        print("[final-eval] FID:", score)

        fname = f"fid_N{fid_num_samples}_seed{seed}.txt"
        with open(os.path.join(master_dir, fname), "w") as ftxt:
            ftxt.write(f"FID: {score}\n")
        
#-------------------------
# yaml file parser
#-------------------------

parser = argparse.ArgumentParser(description='Load hyperparameters from a YAML file.')
parser.add_argument('--hyper_param_config', default = 'config/image_GFM_flowers.yaml',
                    type=str, help='Path to the YAML file')
parser.add_argument('--device', type=str, default=None,
                    help='Device string passed to torch.device (e.g., cuda:0, cpu)')
parser.add_argument('--seed', type=int, default=1103,
                    help='Seed for reproducibility')
args_parsed = parser.parse_args()
with open(args_parsed.hyper_param_config, 'r') as file:
    args_yaml = yaml.safe_load(file)
    print(yaml.dump(args_yaml, default_flow_style=False))

seed = int(args_parsed.seed)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print(f"[INFO] Random seed set to {seed}")

# setup device
device_cfg = args_parsed.device if args_parsed.device is not None else args_yaml.get('device')
if device_cfg is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(device_cfg)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

# Enable TF32 + cuDNN benchmark
#torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def make_warmup_lr(warmup_steps):
    def warmup_lr(step):
        return min(step, warmup_steps) / float(warmup_steps)
    return warmup_lr


if __name__ == '__main__':
    # dataset
    dataname = args_yaml['data']['dataname']
    channels = args_yaml['data']['channels']
    in_shape = (channels, args_yaml['data']['image_shape'][0], args_yaml['data']['image_shape'][1])
    max_train_samples = None if 'ntr' not in args_yaml['data'] else args_yaml['data']['ntr']
    
    root = args_yaml['data'].get('root', './data') #root dir for datasets
    num_workers = int(args_yaml['data'].get('num_workers', 4))
    pin_memory = bool(args_yaml['data'].get('pin_memory', True))

    # result dir
    base_dir = args_yaml['base_dir']
    master_dir = os.path.join(base_dir, args_yaml['save_dir'])
    os.makedirs(master_dir, exist_ok=True)
    print(f'Saving results to: {master_dir}')

    # save printed config to file
    config_txt = yaml.dump(args_yaml, default_flow_style=False)
    with open(os.path.join(master_dir, "config.txt"), "w") as f:
        f.write(config_txt)
        
    # model config
    load_ema = args_yaml['training']['load_ema']
    batch_size = args_yaml['training']['batch_size']
    max_batch_ls = args_yaml['training']['max_batch']
    resume = args_yaml['training']['resume']
    eval_only = args_yaml['training']['eval_only']
    if eval_only:
        print("Running in evaluation only mode")

    # training
    use_grad_clip = True if 'use_grad_clip' not in args_yaml['training'] else args_yaml['training']['use_grad_clip']
    ema_decay = 0.9999 if 'ema_decay' not in args_yaml['training'] else args_yaml['training']['ema_decay']
    batch_push_size = 200 if 'batch_push_size' not in args_yaml['training'] else args_yaml['training']['batch_push_size']

    viz_freq = args_yaml['visualize']['viz_freq']
    
    cal_fid = args_yaml['eval']['cal_fid']
    gen_bs = 200 if 'gen_bs' not in args_yaml['eval'] else args_yaml['eval']['gen_bs']
    fid_num_samples = 5000 if 'fid_num_samples' not in args_yaml['eval'] else args_yaml['eval']['fid_num_samples']
    
    # args_yaml['task'] = args_yaml['save_dir']
    # args_yaml['seed'] = seed

    # model
    L = args_yaml['model']['L'] #N=L+1
    hks = args_yaml['model']['hks']
    hidden_dim, dim_mults = args_yaml['model']['hidden_dim'], args_yaml['model']['unet_dim_mults']
    # attention resolutions: wrapper expects a comma-separated string like "16,8"
    mcfg = args_yaml.get("model", {})
    attn = mcfg.get("attention_resolutions", mcfg.get("attention_resolution", "16"))
    if isinstance(attn, (list, tuple)):
        attention_resolutions = ",".join(str(int(x)) for x in attn)
    else:
        attention_resolutions = str(attn).replace(" ", "")
    
    dropout = float(args_yaml['model']['dropout']) 
    num_res_blocks = int(args_yaml['model']['res_depth']) 
    print(f"Unet: base channels: {hidden_dim}, dim_mults: {dim_mults}, res_depth: {num_res_blocks}, attn_reso: {attention_resolutions}")

    coupling ='dependent' if 'coupling' not in args_yaml['model'] else args_yaml['model']['coupling']
    interp = 'linear' if 'interp' not in args_yaml['model'] else args_yaml['model']['interp']
    # beta t sampling
    beta_cfg = None
    if args_yaml['model'].get('beta_sample_t', False):
        beta_cfg = {"enabled": True, "alpha": args_yaml['model']['alpha'], "beta": args_yaml['model']["beta"]}
        print(f"beta sample enabled: {beta_cfg['alpha']}, {beta_cfg['beta']}")

    #ode
    ode_cfg = args_yaml['ode']
    print("ODE config:")
    print(ode_cfg)

    if eval_only:
        run_final_eval(
            root = root,
            master_dir=master_dir,
            seed=seed,
            L=L,
            in_shape=in_shape,
            hidden_dim=hidden_dim,
            dim_mults=dim_mults,
            attention_resolutions = attention_resolutions,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            ode_cfg=ode_cfg,
            load_ema=load_ema,
            cal_fid=cal_fid,
            fid_num_samples=fid_num_samples,
            gen_bs=gen_bs,
            device=device)
        
        raise SystemExit(0)

    # ---------------------------------------
    # training
    # ---------------------------------------
    # Use AMP (autocast + GradScaler)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # initialize the train_dl for block 1
    train_dl = make_imagenet32_loaders(
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        max_train_samples= max_train_samples, #comment out if use full imagenet
    )

    flow_ls = []  # persistent list of trained blocks
    for l in range(1, L+2):
  
        hk = hks[l-1]
    
        if isinstance(max_batch_ls, int):
            max_batch = max_batch_ls
        else:
            max_batch = max_batch_ls[l-1]

        print(f'####### Block {l} with hk = {hk}')

        # prepare data by pushing when l >1
        if l > 1:
            if len(flow_ls) == l - 1: #check if the length of flow_ls is correct
                # prepare data loader for training l-th block
                push_dl = DataLoader(
                    train_dl.dataset,
                    batch_size = batch_push_size,
                    shuffle=False,      # recommended for reproducibility
                    drop_last=False,    # IMPORTANT: include the last partial batch
                    num_workers=train_dl.num_workers,
                    pin_memory=train_dl.pin_memory,
                    persistent_workers=getattr(train_dl, "persistent_workers", False),
                )
                pool_cpu = push_loader_by_current_prealloc(flow_ls[-1], push_dl, 
                                                           device=device, out_dtype=torch.float16)
                print(f"[push] expected N={len(push_dl.dataset)} got pool_cpu.shape = {tuple(pool_cpu.shape)}")
                # then rewrite train_dl
                train_dl = make_pool_loader(pool_cpu, batch_size=batch_size, 
                                            shuffle=True, num_workers=0, 
                                            pin_memory=pin_memory, drop_last=True)
            else:
                raise RuntimeError(f"[block {l}] flow_ls length mismatch: "
                                   f"expected {l-1} previous blocks, got {len(flow_ls)}.\n"
                                   f"Check resume logic, load_prev_blocks(), or missing flow_ls.append().")
                 
        # initial flow from scratch, and ema = flow
        flow = CNF(in_shape, hidden_dim,  dim_mults, attention_resolutions, num_res_blocks, dropout, ode_cfg).to(device)
        print(f'Number of parameters: {count_params(flow):.3f}k')
        ema = deepcopy(flow).to(device) #keep a copy for ema, both flow and ema are on device
        ema.eval()  
        #warm start if l > 1 and asked so
        if l > 1 and args_yaml['training']['warm_start']:
            flow.load_state_dict(flow_ls[-1].state_dict())
            ema.load_state_dict(flow_ls[-1].state_dict())  # ema same as flow when warm starting
                   
        # loss, opt, lr
        loss_fn = InterFlowLoss(flow, beta_cfg=beta_cfg, interp= interp).to(device)
        optimizer = torch.optim.Adam(flow.parameters(), lr=float(args_yaml['training']['lr']))
        warmup_steps = args_yaml['training']['warmup']
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                      lr_lambda= make_warmup_lr(warmup_steps))
        start_batch = 0
        loss_ls = []

        # load (flow, ema) from ckpt if resume
        if resume: 
            resume_path = find_block_ckpt(master_dir, l, seed) 
            if resume_path is not None:
                sdict = torch.load(resume_path, map_location="cpu", weights_only= False)
                flow.load_state_dict(sdict['state_dict']) #load flow
                if 'ema_state_dict' in sdict:
                    ema.load_state_dict(sdict['ema_state_dict']) #load ema
                else:
                    ema = deepcopy(flow).to(device) #if no ema, then ema is flow
                start_batch = sdict['batch']
                loss_ls = sdict['loss_ls']  
                if start_batch < max_batch: #only load optimizer and scheduler if training of this block is not finished
                    optimizer.load_state_dict(sdict['optimizer'])
                    scheduler.load_state_dict(sdict['scheduler'])
                print(f"Loading {resume_path} with {'ema_state_dict' if 'ema_state_dict' in sdict else 'state_dict'} |"
                        f"start batch: {start_batch}")
            else: 
                print(f"[block {l}] resume requested but no checkpoint found; starting from scratch.")
            
        # count peak memory only for training
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        
        train_it = infinite_loader(train_dl)
        indep_it = infinite_loader(train_dl)   # independent stream

        last_batch = start_batch - 1
        for batch in tqdm(range(start_batch, max_batch), ncols=88):
            last_batch = batch
            # sample x0 
            x0 = unwrap_x(next(train_it)).to(device, dtype=torch.float32, non_blocking=True)
                            # (B,C,H,W) in [-1,1] on CPU

            # sample x1
            z = torch.randn_like(x0, dtype=torch.float32)
            if l == L+1:
                x1 = z #always independent coupling
            else:
                B = x0.shape[0]
                if coupling == "dependent":
                    # x1 depends on x0 (your current behavior)
                    x0p = x0
                elif coupling == "shuffled":
                    # x1 depends on a shuffled version of x0
                    perm = torch.randperm(B, device=device)
                    x0p = x0[perm]
                elif coupling == "independent":
                    # x1 depends on an independent batch from the dataset
                    x0p = unwrap_x(next(indep_it)).to(device, dtype=torch.float32, non_blocking=True)         
                else:
                    raise ValueError(f"Unknown coupling mode: {coupling}")
                shrink_ou = math.exp(-hk)
                sigma_ou = math.sqrt(max(0.0, 1.0 - shrink_ou **2))
                x1 = shrink_ou*x0p + sigma_ou*z # OU process
        
            # # update flow and ema
            # optimizer.zero_grad(set_to_none=True)
            # with autocast("cuda", enabled=(device.type == "cuda")):
            #     loss_x = loss_fn(x0, x1)
            # loss_ls.append(float(loss_x.item()))
            # scaler.scale(loss_x).backward()
            # scaler.step(optimizer)
            # scaler.update() 
            # scheduler.step()
            
            # if batch < warmup_steps:
            #     ema_decay = 0.0
            # else:
            #     ema_decay = 0.9999
            # update_ema(ema, flow, decay=ema_decay) #effective window 10k steps

            # update flow and ema
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                loss_x = loss_fn(x0, x1)    
            loss_ls.append(float(loss_x.item()))
            scaler.scale(loss_x).backward()
            
            did_step = True
            total_norm = None
            if use_grad_clip: #if not using grad_clip, then keep updaing even grads are inf
                scaler.unscale_(optimizer) # grad norm clipping (AMP-safe)
                total_norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
                if batch % 1000 == 0:
                    print(f"[grad-norm] {total_norm:.3f}")
            if total_norm is not None and (not torch.isfinite(total_norm)):
                did_step = False
                optimizer.zero_grad(set_to_none=True)
                scaler.update()              # adjust scaler only
            else:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()             # only step scheduler if optimizer stepped       
            if did_step:
                decay = 0.0 if batch < warmup_steps else ema_decay
                update_ema(ema, flow, decay=decay)
                  
            if (batch % viz_freq == 0 or batch == max_batch-1) and batch > 0:
                # print memory
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
                sdict_path = os.path.join(master_dir, f'state_dict_block_{l}_seed{seed}.pt')
                save_path = sdict_path if batch == max_batch-1 else sdict_path.replace('.pt', f'_latest.pt')
                #save_path = sdict_path if batch == max_batch-1 else sdict_path.replace('.pt', f'_{batch}.pt')
                sdict = {'state_dict': flow.state_dict(),
                        'ema_state_dict': ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'batch': last_batch+1,
                        'loss_ls': loss_ls}
                torch.save(sdict, save_path)
                
        print(f"Done with training block {l}!")
        #attach the official model for this block to the flow_ls list, flow or ema
        blk = deepcopy(ema if load_ema else flow).to(device).eval()
        flow_ls.append(blk)
        # print peak memory for this block
        if device.type == "cuda":
            print(f"[block {l} GPU peak] {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
        
        # free the per-block training models
        del flow, ema
        optimizer = None
        scheduler = None
        loss_fn = None
        torch.cuda.empty_cache()

    print("All blocks finished. Running final evaluation...")
    run_final_eval(
        root=root,
        master_dir=master_dir,
        seed=seed,
        L=L,
        in_shape=in_shape,
        hidden_dim=hidden_dim,
        dim_mults=dim_mults,
        attention_resolutions=attention_resolutions,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        ode_cfg=ode_cfg,
        load_ema=load_ema,
        cal_fid=cal_fid,
        fid_num_samples=fid_num_samples,
        gen_bs=gen_bs,
        device=device)

