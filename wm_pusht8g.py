"""Windows-friendly PushT: cosine diffusion, periodic conditioning, EMA and action chunks.
Educational small-scale reproduction scaffold; NOT the official Diffusion Policy implementation.
Run `python wm_pusht8g.py --help` for supported commands.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import urllib.request
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

DATA_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "runs_improved"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_of(choice: str) -> torch.device:
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: use `python ... check`, then verify NVIDIA driver/PyTorch CUDA installation")
    return torch.device(choice)


def check_env() -> None:
    print("Python torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    try:
        import gymnasium as gym  # noqa: F401
        import gym_pusht  # noqa: F401
        env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
        state, info = env.reset(seed=42)
        print("PushT state shape:", np.asarray(state).shape)
        _, reward, done, trunc, info = env.step(np.array([256.0, 300.0], dtype=np.float32))
        print("PushT step OK, reward:", round(float(reward), 4))
        env.close()
    except ImportError as ex:
        print("PushT not installed (selftest can still run):", ex)


def locate_zarr() -> Path:
    for p in DATA_DIR.rglob(".zgroup"):
        root = p.parent
        if (root / "data" / "state" / ".zarray").exists():
            return root
    raise FileNotFoundError("Cannot find Zarr with data/state. First run `python wm_pusht8g.py download`.")


def download() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    try:
        found = locate_zarr()
        print("Already downloaded:", found)
        return
    except FileNotFoundError:
        pass
    z = DATA_DIR / "pusht.zip"
    if not z.exists():
        print("Downloading official 29MB PushT demonstrations from:", DATA_URL)
        urllib.request.urlretrieve(DATA_URL, z)
    print("Extracting...")
    with zipfile.ZipFile(z) as f:
        f.extractall(DATA_DIR)
    print("Located:", locate_zarr())


def load_arrays(synthetic: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if synthetic:
        rng = np.random.default_rng(7)
        lengths = [20] * 10
        states, actions = [], []
        for n in lengths:
            state = rng.uniform(0.2, 0.8, size=(5,)).astype(np.float32)
            for _ in range(n):
                a = np.clip(state[:2] + rng.normal(0, 0.1, 2), 0, 1).astype(np.float32)
                states.append(state.copy())
                actions.append(a * 512)
                state = np.clip(state + np.r_[0.03 * (a - state[:2]), np.zeros(3)], 0, 1).astype(np.float32)
        states = np.asarray(states, dtype=np.float32) * 512
        states[:, 4] = 0.2
        return states, np.asarray(actions, dtype=np.float32), np.cumsum(lengths).astype(np.int64)
    import zarr
    root = zarr.open(str(locate_zarr()), mode="r")
    state = np.asarray(root["data/state"][:], dtype=np.float32)
    action = np.asarray(root["data/action"][:], dtype=np.float32)
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if state.ndim != 2 or state.shape[1] != 5 or action.shape != (len(state), 2):
        raise ValueError(f"Unexpected data shape: state={state.shape}, action={action.shape}")
    if ends[-1] != len(state):
        raise ValueError("episode_ends does not match data length")
    return state, action, ends


def norm_s(x):
    # Keep state coords (pixels) and angle on roughly comparable scales.
    if isinstance(x, torch.Tensor):
        scale = x.new_tensor([256, 256, 256, 256, math.pi])
    else:
        scale = np.array([256, 256, 256, 256, math.pi], dtype=np.float32)
    return x / scale - 1.0


def denorm_s(x):
    if isinstance(x, torch.Tensor):
        scale = x.new_tensor([256, 256, 256, 256, math.pi])
    else:
        scale = np.array([256, 256, 256, 256, math.pi], dtype=np.float32)
    return (x + 1.0) * scale


def norm_a(x):
    return x / 256.0 - 1.0


def denorm_a(x):
    return (x + 1.0) * 256.0


class WindowDataset(Dataset):
    def __init__(self, states, actions, ends, split="train", obs_len=2, horizon=8, seed=42):
        self.states = norm_s(states).astype(np.float32)
        self.actions = norm_a(actions).astype(np.float32)
        self.obs_len = obs_len
        self.horizon = horizon
        n = len(ends)
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        train_set = set(order[:max(1, int(n * 0.9))])
        self.indices = []
        starts = np.r_[0, ends[:-1]]
        for ep, (start, end) in enumerate(zip(starts, ends)):
            use = ep in train_set
            if (split == "train") == use:
                for i in range(int(start), int(end)):
                    self.indices.append((i, int(start), int(end)))
        if not self.indices:
            raise RuntimeError(f"Empty dataset split: {split}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        t, start, end = self.indices[index]
        o_idx = np.clip(np.arange(t - self.obs_len + 1, t + 1), start, end - 1)
        a_idx = np.clip(np.arange(t, t + self.horizon), start, end - 1)
        return (torch.from_numpy(self.states[o_idx]), torch.from_numpy(self.actions[a_idx]))


def policy_features(obs, geometric=True):
    """Encode the T outline in world coordinates and the pusher in block coordinates."""
    angle = (obs[..., 4:5] + 1) * math.pi
    cs, sn = angle.cos(), angle.sin()
    if not geometric:
        return torch.cat((obs[..., :4], sn, cs), dim=-1)
    # Outline vertices of gym-pusht's two rectangles, relative to body origin.
    vertices = obs.new_tensor([[-60, 0], [60, 0], [60, 30], [15, 30],
                               [15, 120], [-15, 120], [-15, 30], [-60, 30]]) / 256
    vx, vy = vertices[:, 0], vertices[:, 1]
    xs = obs[..., 2:3] + cs * vx - sn * vy
    ys = obs[..., 3:4] + sn * vx + cs * vy
    keypoints = torch.stack((xs, ys), dim=-1).flatten(-2)
    delta = obs[..., :2] - obs[..., 2:4]
    local_x = cs * delta[..., :1] + sn * delta[..., 1:2]
    local_y = -sn * delta[..., :1] + cs * delta[..., 1:2]
    return torch.cat((obs[..., :2], keypoints, local_x, local_y), dim=-1)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim):
        super().__init__()
        groups = math.gcd(out_channels, 8)
        self.first = nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, padding=1),
                                   nn.GroupNorm(groups, out_channels), nn.SiLU())
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 2 * out_channels))
        self.second = nn.Sequential(nn.Conv1d(out_channels, out_channels, 3, padding=1),
                                    nn.GroupNorm(groups, out_channels), nn.SiLU())
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, condition):
        scale, shift = self.film(condition).unsqueeze(-1).chunk(2, dim=1)
        h = self.first(x) * (1 + scale) + shift
        return self.second(h) + self.skip(x)


class TemporalUNet(nn.Module):
    """Small conditional 1D U-Net; convolutions operate over future action time."""
    def __init__(self, width, condition_dim):
        super().__init__()
        self.enc1 = TemporalBlock(2, width, condition_dim)
        self.down1 = nn.Conv1d(width, 2 * width, 3, stride=2, padding=1)
        self.enc2 = TemporalBlock(2 * width, 2 * width, condition_dim)
        self.down2 = nn.Conv1d(2 * width, 4 * width, 3, stride=2, padding=1)
        self.mid1 = TemporalBlock(4 * width, 4 * width, condition_dim)
        self.mid2 = TemporalBlock(4 * width, 4 * width, condition_dim)
        self.dec2 = TemporalBlock(6 * width, 2 * width, condition_dim)
        self.dec1 = TemporalBlock(3 * width, width, condition_dim)
        self.output = nn.Conv1d(width, 2, 1)

    def forward(self, action, condition):
        s1 = self.enc1(action.transpose(1, 2), condition)
        s2 = self.enc2(self.down1(s1), condition)
        mid = self.mid2(self.mid1(self.down2(s2), condition), condition)
        up2 = F.interpolate(mid, size=s2.shape[-1], mode="nearest")
        up2 = self.dec2(torch.cat((up2, s2), dim=1), condition)
        up1 = F.interpolate(up2, size=s1.shape[-1], mode="nearest")
        return self.output(self.dec1(torch.cat((up1, s1), dim=1), condition)).transpose(1, 2)


class Denoiser(nn.Module):
    def __init__(self, hidden=64, horizon=16, obs_len=2, diffusion_steps=100,
                 architecture="unet", schedule="cosine", prediction_type="v_prediction"):
        super().__init__()
        if architecture not in {"legacy", "residual", "geometric", "unet"}:
            raise ValueError(f"Unknown architecture: {architecture}")
        if prediction_type not in {"epsilon", "v_prediction"} or schedule not in {"linear", "cosine"}:
            raise ValueError("Unsupported diffusion prediction type or schedule")
        self.horizon, self.obs_len = horizon, obs_len
        self.steps = diffusion_steps
        self.architecture, self.schedule = architecture, schedule
        self.prediction_type = prediction_type
        if architecture == "legacy":
            in_dim = 5 * obs_len + 2 * horizon + 32
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                     nn.SiLU(), nn.Linear(hidden, horizon * 2))
        elif architecture == "unet":
            condition_dim = hidden * 4
            self.condition = nn.Sequential(nn.Linear(20 * obs_len + 32, condition_dim),
                                           nn.SiLU(), nn.Linear(condition_dim, condition_dim))
            self.temporal = TemporalUNet(hidden, condition_dim)
        else:
            # sin/cos removes the artificial discontinuity at angle 0 / 2*pi.
            feature_dim = 20 if architecture == "geometric" else 6
            self.condition = nn.Sequential(nn.Linear(feature_dim * obs_len + 32, hidden), nn.SiLU(),
                                           nn.Linear(hidden, hidden))
            self.input = nn.Linear(2 * horizon, hidden)
            self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(4)])
            self.output = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, horizon * 2))
        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        else:
            grid = torch.linspace(0, 1, diffusion_steps + 1, dtype=torch.float64)
            curve = torch.cos((grid + 0.008) / 1.008 * math.pi / 2).square()
            betas = (1 - curve[1:] / curve[:-1]).clamp(0.0001, 0.999).float()
        self.register_buffer("alpha_bars", torch.cumprod(1 - betas, dim=0))

    def forward(self, obs, noisy_action, t):
        half = 16
        freqs = torch.exp(torch.arange(half, device=t.device).float() * (-math.log(10000) / (half - 1)))
        time = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        temb = torch.cat((time.sin(), time.cos()), dim=1)
        if self.architecture == "legacy":
            inp = torch.cat((obs.flatten(1), noisy_action.flatten(1), temb), dim=1)
            return self.net(inp).view(-1, self.horizon, 2)
        features = policy_features(obs, geometric=self.architecture in {"geometric", "unet"})
        cond = self.condition(torch.cat((features.flatten(1), temb), dim=1))
        if self.architecture == "unet":
            return self.temporal(noisy_action, cond)
        x = self.input(noisy_action.flatten(1))
        for block in self.blocks:
            x = x + block(x + cond)
        return self.output(x + cond).view(-1, self.horizon, 2)

    def loss(self, obs, act):
        b = obs.shape[0]
        t = torch.randint(self.steps, (b,), device=obs.device)
        eps = torch.randn_like(act)
        ab = self.alpha_bars[t].view(-1, 1, 1)
        noisy = ab.sqrt() * act + (1 - ab).sqrt() * eps
        target = eps if self.prediction_type == "epsilon" else ab.sqrt() * eps - (1 - ab).sqrt() * act
        return F.mse_loss(self(obs, noisy, t), target)

    @torch.no_grad()
    def sample(self, obs, n_inference_steps=32):
        # Deterministic DDIM (eta=0), not identical to paper's official sampler.
        if not 2 <= n_inference_steps <= self.steps:
            raise ValueError(f"sample_steps must be between 2 and {self.steps}")
        b = obs.shape[0]
        x = torch.randn((b, self.horizon, 2), device=obs.device)
        ts = torch.linspace(self.steps - 1, 0, n_inference_steps, device=obs.device).round().long().tolist()
        for j, ti in enumerate(ts):
            t = torch.full((b,), ti, device=obs.device, dtype=torch.long)
            prediction = self(obs, x, t)
            ab = self.alpha_bars[ti]
            if self.prediction_type == "epsilon":
                x0 = (x - (1 - ab).sqrt() * prediction) / ab.sqrt()
            else:
                # Velocity prediction avoids division by near-zero terminal SNR.
                x0 = ab.sqrt() * x - (1 - ab).sqrt() * prediction
            # Near-zero terminal SNR amplifies epsilon error. Bound the clean
            # action estimate at EVERY step, not just after the entire rollout.
            x0 = x0.clamp(-1, 1)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            next_t = ts[j + 1] if j + 1 < len(ts) else -1
            ap = self.alpha_bars[next_t] if next_t >= 0 else torch.ones_like(ab)
            x = ap.sqrt() * x0 + (1 - ap).sqrt() * eps
        return x.clamp(-1, 1)


class WorldMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, 5))

    def forward(self, state, action):
        return state + self.net(torch.cat([state, action], -1))


class TransitionDataset(Dataset):
    def __init__(self, states, actions, ends, split="train", seed=42):
        n = len(ends)
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        train_set = set(order[:max(1, int(n * 0.9))])
        starts = np.r_[0, ends[:-1]]
        rows = []
        for ep, (start, end) in enumerate(zip(starts, ends)):
            if (ep in train_set) == (split == "train"):
                rows.extend(range(int(start), int(end) - 1))
        self.s = torch.from_numpy(norm_s(states[rows]).astype(np.float32))
        self.a = torch.from_numpy(norm_a(actions[rows]).astype(np.float32))
        next_s = norm_s(states[np.asarray(rows) + 1]).astype(np.float32)
        self.sp = torch.from_numpy(next_s)
        if len(self.s) == 0:
            raise RuntimeError("Empty transition dataset")

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.s[i], self.a[i], self.sp[i]


def train_policy(args):
    states, actions, ends = load_arrays(args.synthetic)
    train_set = WindowDataset(states, actions, ends, "train", horizon=args.horizon, seed=args.seed)
    val_set = WindowDataset(states, actions, ends, "val", horizon=args.horizon, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dev = device_of(args.device)
    model = Denoiser(hidden=args.hidden, horizon=args.horizon, architecture=args.architecture).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    warmup = min(500, max(1, args.steps // 10))
    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_factor)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT_DIR / "training_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best_val = float("inf")
    print(f"Train windows={len(train_set)}, validation windows={len(val_set)}, "
          f"terminal alpha_bar={model.alpha_bars[-1].item():.8f}", flush=True)
    it = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            o, a = next(it)
        except StopIteration:
            it = iter(train_loader)
            o, a = next(it)
        model.train()
        optim.zero_grad(set_to_none=True)
        loss = model.loss(o.to(dev), a.to(dev))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()
        decay = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for average, param in zip(ema.parameters(), model.parameters()):
                average.lerp_(param, 1 - decay)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            # Same validation noise and all held-out episodes at each check.
            devices = [torch.cuda.current_device()] if dev.type == "cuda" else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                torch.manual_seed(args.seed + 100000)
                total = 0.0
                for vo, va in val_loader:
                    total += ema.loss(vo.to(dev), va.to(dev)).item() * len(vo)
                vl = total / len(val_set)
            print(f"policy step={step:6d} train_loss={loss.item():.5f} ema_val_loss={vl:.5f}", flush=True)
            with metrics_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps({"step": step, "train_loss": loss.item(),
                                      "ema_val_loss": vl, "lr": optim.param_groups[0]["lr"]}) + "\n")
            checkpoint = {"model": ema.state_dict(), "horizon": args.horizon,
                          "hidden": args.hidden, "seed": args.seed, "steps": step,
                          "version": 3, "schedule": "cosine", "architecture": model.architecture,
                          "prediction_type": model.prediction_type,
                          "diffusion_steps": model.steps, "val_loss": vl,
                          "synthetic": args.synthetic, "batch_size": args.batch_size,
                          "learning_rate": args.lr, "obs_len": model.obs_len}
            # policy.pt is the best validation EMA; retain the final EMA too.
            save_checkpoint(checkpoint, OUT_DIR / "policy_last.pt")
            if vl < best_val:
                best_val = vl
                save_checkpoint(checkpoint, OUT_DIR / "policy.pt")
    print("Saved best validation EMA:", OUT_DIR / "policy.pt", "loss=", best_val)


def train_world(args):
    states, actions, ends = load_arrays(args.synthetic)
    tr = TransitionDataset(states, actions, ends, "train", seed=args.seed)
    va = TransitionDataset(states, actions, ends, "val", seed=args.seed)
    loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    vloader = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dev = device_of(args.device)
    model = WorldMLP(hidden=args.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    it = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            s, a, sp = next(it)
        except StopIteration:
            it = iter(loader)
            s, a, sp = next(it)
        s, a, sp = s.to(dev), a.to(dev), sp.to(dev)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(s, a), sp)
        loss.backward()
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                vs, vact, vsp = next(iter(vloader))
                val_loss = F.mse_loss(model(vs.to(dev), vact.to(dev)), vsp.to(dev)).item()
            print(f"world  step={step:6d} train_loss={loss.item():.6f} val_loss={val_loss:.6f}", flush=True)
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "world.pt"
    torch.save({"model": model.cpu().state_dict(), "hidden": args.hidden,
                "seed": args.seed, "steps": args.steps}, path)
    print("Saved", path)


def save_checkpoint(checkpoint, path):
    """Readers see either the old or new complete checkpoint, never a partial write."""
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_policy(dev, path=None):
    path = Path(path) if path else OUT_DIR / "policy.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    legacy = ckpt.get("version", 1) < 3
    if legacy:
        print("Legacy checkpoint: architecture and training schedule preserved; retrain for the full fix.")
    if ckpt.get("synthetic"):
        print("WARNING: synthetic-debug checkpoint; not a trained PushT policy.")
    model = Denoiser(hidden=ckpt["hidden"], horizon=ckpt["horizon"],
                     obs_len=ckpt.get("obs_len", 2),
                     diffusion_steps=len(ckpt["model"]["alpha_bars"]),
                     architecture="legacy" if legacy else ckpt["architecture"],
                     schedule=ckpt.get("schedule", "linear"),
                     prediction_type=ckpt.get("prediction_type", "epsilon")).to(dev)
    model.load_state_dict(ckpt["model"])
    model.checkpoint_metadata = {key: value for key, value in ckpt.items() if key != "model"}
    return model.eval()


def load_world(dev, path=None):
    path = Path(path) if path else OUT_DIR / "world.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = WorldMLP(hidden=ckpt["hidden"]).to(dev)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def cost_from_pred(sn):
    s = denorm_s(sn)
    dxy = ((s[..., 2:4] - s.new_tensor([256., 256.])) ** 2).sum(-1) / (256. ** 2)
    dtheta = torch.atan2((s[..., 4] - math.pi / 4).sin(), (s[..., 4] - math.pi / 4).cos())
    return dxy + 0.15 * dtheta.square()


@torch.no_grad()
def select_chunk(policy, world, history, mode, candidates, sample_steps, action_steps=4):
    dev = next(policy.parameters()).device
    hist = torch.from_numpy(norm_s(np.stack(history)[None]).astype(np.float32)).to(dev)
    if mode == "policy":
        actions = policy.sample(hist, sample_steps)[0]
        return denorm_a(actions).cpu().numpy()
    opts = policy.sample(hist.repeat(candidates, 1, 1), sample_steps)
    if mode == "sample4":
        return denorm_a(opts[0]).cpu().numpy()
    if world is None:
        raise RuntimeError("world mode requires a trained world.pt")
    current = hist[:, -1].repeat(candidates, 1)
    # One-step dynamics rolled out over a few candidate actions.
    for i in range(min(action_steps, opts.shape[1])):
        current = world(current, opts[:, i])
    scores = cost_from_pred(current)
    best = int(torch.argmin(scores).item())
    return denorm_a(opts[best]).cpu().numpy()


def select_action(policy, world, history, mode, candidates, sample_steps):
    return select_chunk(policy, world, history, mode, candidates, sample_steps)[0]


class ChunkController:
    """Shared deployment path for evaluation and visualization; reset each episode."""
    def __init__(self, policy, world=None, mode="policy", candidates=4,
                 sample_steps=32, action_steps=4):
        if not 1 <= action_steps <= policy.horizon:
            raise ValueError(f"action_steps must be between 1 and horizon={policy.horizon}")
        if candidates < 1 or not 2 <= sample_steps <= policy.steps:
            raise ValueError(f"candidates must be positive; sample_steps must be 2..{policy.steps}")
        self.policy, self.world, self.mode = policy, world, mode
        self.candidates, self.sample_steps, self.action_steps = candidates, sample_steps, action_steps
        self.reset()

    def reset(self):
        self.history, self.pending = [], []

    def act(self, state):
        state = np.asarray(state, dtype=np.float32).copy()
        self.history = (self.history + [state])[-self.policy.obs_len:]
        while len(self.history) < self.policy.obs_len:
            self.history.insert(0, state.copy())
        if not self.pending:
            chunk = select_chunk(self.policy, self.world, self.history, self.mode,
                                 self.candidates, self.sample_steps, self.action_steps)
            self.pending = list(np.clip(chunk[:self.action_steps], 0, 512).astype(np.float32))
        return self.pending.pop(0)


def evaluate(args):
    import gymnasium as gym
    import gym_pusht  # noqa F401
    dev = device_of(args.device)
    policy = load_policy(dev, args.checkpoint)
    world = load_world(dev, args.world_checkpoint) if args.mode == "world" else None
    controller = ChunkController(policy, world, args.mode, args.candidates, args.sample_steps, args.action_steps)
    rewards, successes = [], []
    with gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array",
                  max_episode_steps=args.max_steps) as env:
        for ep in range(args.episodes):
            seed_all(args.seed + ep)
            state, info = env.reset(seed=args.seed + ep)
            controller.reset()
            best_reward = 0.0
            success = False
            for step in range(args.max_steps):
                act = controller.act(state)
                state, reward, terminated, truncated, info = env.step(act)
                best_reward = max(best_reward, float(reward))
                success = success or bool(info.get("is_success", terminated))
                if terminated or truncated:
                    break
            rewards.append(best_reward)
            successes.append(int(success))
            print(f"episode {ep+1}/{args.episodes}: success={success}, max_reward={best_reward:.3f}", flush=True)
    print(f"RESULT mode={args.mode}: success_rate={np.mean(successes):.3f}, max_reward_mean={np.mean(rewards):.3f}, n={args.episodes}")
    print("WARNING: these are local results of a light-weight implementation, NOT paper benchmark scores.")
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"checkpoint": str(args.checkpoint or OUT_DIR / "policy.pt"),
            "training": policy.checkpoint_metadata,
            "mode": args.mode, "device": args.device, "seed": args.seed, "episodes": args.episodes,
            "max_steps": args.max_steps, "action_steps": args.action_steps,
            "sample_steps": args.sample_steps, "success_rate": float(np.mean(successes)),
            "max_reward_mean": float(np.mean(rewards)), "successes": successes,
            "max_rewards": rewards}, indent=2), encoding="utf-8")


def smoke(args):
    print("Synthetic selftest (no robot demonstrations and no gym required):")
    dev = device_of(args.device)
    states, actions, ends = load_arrays(synthetic=True)
    ds = WindowDataset(states, actions, ends, "train")
    obs, act = next(iter(DataLoader(ds, batch_size=4, num_workers=0)))
    policy = Denoiser(hidden=64, horizon=8).to(dev)
    loss = policy.loss(obs.to(dev), act.to(dev))
    loss.backward()
    sample = policy.sample(obs.to(dev), 4)
    tr = TransitionDataset(states, actions, ends, "train")
    s, a, sp = next(iter(DataLoader(tr, batch_size=4, num_workers=0)))
    wm = WorldMLP(hidden=32).to(dev)
    wloss = F.mse_loss(wm(s.to(dev), a.to(dev)), sp.to(dev))
    wloss.backward()
    assert sample.shape == (4, 8, 2)
    print("PASS: policy forward/backward and sample", tuple(sample.shape), "loss", float(loss.detach()))
    print("PASS: world forward/backward", "loss", float(wloss.detach()))


def main():
    global OUT_DIR
    p = argparse.ArgumentParser(description="Windows 8GB PushT world-model + diffusion policy learning project")
    sub = p.add_subparsers(dest="command", required=True)
    for c in ["check", "download", "inspect", "smoke", "train-policy", "train-world", "eval"]:
        q = sub.add_parser(c)
        if c in ("smoke", "train-policy", "train-world", "eval"):
            q.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
            q.add_argument("--cpu-threads", type=int, default=1, help="CPU tensor threads; small state models need few")
            q.add_argument("--seed", type=int, default=42)
            q.add_argument("--out-dir", type=Path, default=OUT_DIR)
        if c in ("train-policy", "train-world"):
            q.add_argument("--steps", type=int, default=30000 if c == "train-policy" else 10000)
            q.add_argument("--batch-size", type=int, default=128)
            q.add_argument("--hidden", type=int, default=64 if c == "train-policy" else 128)
            q.add_argument("--lr", type=float, default=1e-4)
            q.add_argument("--log-every", type=int, default=500)
            q.add_argument("--synthetic", action="store_true", help="Only use for debugging; NOT real training")
            if c == "train-policy":
                q.add_argument("--horizon", type=int, default=16)
                q.add_argument("--architecture", choices=["unet", "geometric"], default="unet")
        if c == "eval":
            q.add_argument("--mode", choices=["policy", "sample4", "world"], default="policy")
            q.add_argument("--episodes", type=int, default=5)
            q.add_argument("--max-steps", type=int, default=300)
            q.add_argument("--candidates", type=int, default=4)
            q.add_argument("--sample-steps", type=int, default=32)
            q.add_argument("--action-steps", type=int, default=4)
            q.add_argument("--checkpoint", type=Path)
            q.add_argument("--world-checkpoint", type=Path)
            q.add_argument("--report", type=Path)
    args = p.parse_args()
    for name in ("steps", "batch_size", "hidden", "log_every", "horizon", "episodes", "max_steps", "candidates", "action_steps", "cpu_threads"):
        if hasattr(args, name) and getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "lr") and args.lr <= 0:
        p.error("--lr must be positive")
    if hasattr(args, "out_dir"):
        OUT_DIR = args.out_dir
        if getattr(args, "synthetic", False):
            OUT_DIR = args.out_dir / "synthetic_debug"
    if hasattr(args, "seed"):
        seed_all(args.seed)
        torch.set_num_threads(args.cpu_threads)
    if args.command == "check":
        check_env()
    elif args.command == "download":
        download()
    elif args.command == "inspect":
        s, a, e = load_arrays()
        print("state:", s.shape, "action:", a.shape, "episodes:", len(e), "first state:", s[0])
    elif args.command == "smoke":
        smoke(args)
    elif args.command == "train-policy":
        train_policy(args)
    elif args.command == "train-world":
        train_world(args)
    elif args.command == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
