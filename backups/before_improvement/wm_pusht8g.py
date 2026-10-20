"""Windows-friendly, lightweight PushT experiment: diffusion action policy + action-conditioned dynamics.
Educational small-scale reproduction scaffold; NOT the official Diffusion Policy implementation.
Run `python wm_pusht8g.py --help` for supported commands.
"""
from __future__ import annotations

import argparse
import math
import os
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
DATA_DIR = Path("data")
OUT_DIR = Path("runs")


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


class Denoiser(nn.Module):
    def __init__(self, hidden=256, horizon=8, obs_len=2, diffusion_steps=32):
        super().__init__()
        self.horizon, self.obs_len = horizon, obs_len
        self.steps = diffusion_steps
        in_dim = 5 * obs_len + 2 * horizon + 32
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, horizon * 2))
        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        self.register_buffer("alpha_bars", torch.cumprod(1 - betas, dim=0))

    def forward(self, obs, noisy_action, t):
        half = 16
        freqs = torch.exp(torch.arange(half, device=t.device).float() * (-math.log(10000) / (half - 1)))
        time = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        temb = torch.cat((time.sin(), time.cos()), dim=1)
        inp = torch.cat((obs.flatten(1), noisy_action.flatten(1), temb), dim=1)
        return self.net(inp).view(-1, self.horizon, 2)

    def loss(self, obs, act):
        b = obs.shape[0]
        t = torch.randint(self.steps, (b,), device=obs.device)
        eps = torch.randn_like(act)
        ab = self.alpha_bars[t].view(-1, 1, 1)
        noisy = ab.sqrt() * act + (1 - ab).sqrt() * eps
        return F.mse_loss(self(obs, noisy, t), eps)

    @torch.no_grad()
    def sample(self, obs, n_inference_steps=8):
        # Deterministic DDIM (eta=0), not identical to paper's official sampler.
        b = obs.shape[0]
        x = torch.randn((b, self.horizon, 2), device=obs.device)
        ts = torch.linspace(self.steps - 1, 0, n_inference_steps, device=obs.device).round().long().tolist()
        for j, ti in enumerate(ts):
            t = torch.full((b,), ti, device=obs.device, dtype=torch.long)
            eps = self(obs, x, t)
            ab = self.alpha_bars[ti]
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
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
    train_set = WindowDataset(states, actions, ends, "train", horizon=args.horizon)
    val_set = WindowDataset(states, actions, ends, "val", horizon=args.horizon)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dev = device_of(args.device)
    model = Denoiser(hidden=args.hidden, horizon=args.horizon).to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
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
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vo, va = next(iter(val_loader))
                vl = model.loss(vo.to(dev), va.to(dev)).item()
            print(f"policy step={step:6d} train_loss={loss.item():.5f} val_loss={vl:.5f}", flush=True)
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "policy.pt"
    torch.save({"model": model.cpu().state_dict(), "horizon": args.horizon,
                "hidden": args.hidden, "seed": args.seed, "steps": args.steps}, path)
    print("Saved", path)


def train_world(args):
    states, actions, ends = load_arrays(args.synthetic)
    tr = TransitionDataset(states, actions, ends, "train", seed=args.seed)
    va = TransitionDataset(states, actions, ends, "val", seed=args.seed)
    loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
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


def load_policy(dev):
    path = OUT_DIR / "policy.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = Denoiser(hidden=ckpt["hidden"], horizon=ckpt["horizon"]).to(dev)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def load_world(dev):
    path = OUT_DIR / "world.pt"
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
def select_action(policy, world, history, mode, candidates, sample_steps):
    dev = next(policy.parameters()).device
    hist = torch.from_numpy(norm_s(np.stack(history)[None]).astype(np.float32)).to(dev)
    if mode == "policy":
        actions = policy.sample(hist, sample_steps)[0]
        return denorm_a(actions[0]).cpu().numpy()
    opts = policy.sample(hist.repeat(candidates, 1, 1), sample_steps)
    if mode == "sample4":
        return denorm_a(opts[0, 0]).cpu().numpy()
    if world is None:
        raise RuntimeError("world mode requires a trained world.pt")
    current = hist[:, -1].repeat(candidates, 1)
    scores = []
    # One-step dynamics rolled out over a few candidate actions.
    for i in range(min(4, opts.shape[1])):
        current = world(current, opts[:, i])
    scores = cost_from_pred(current)
    best = int(torch.argmin(scores).item())
    return denorm_a(opts[best, 0]).cpu().numpy()


def evaluate(args):
    import gymnasium as gym
    import gym_pusht  # noqa F401
    dev = device_of(args.device)
    policy = load_policy(dev)
    world = load_world(dev) if args.mode == "world" else None
    rewards, successes = [], []
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    for ep in range(args.episodes):
        seed_all(args.seed + ep)
        state, info = env.reset(seed=args.seed + ep)
        history = [np.asarray(state, dtype=np.float32)] * 2
        best_reward = 0.0
        success = False
        for step in range(args.max_steps):
            act = select_action(policy, world, history, args.mode, args.candidates, args.sample_steps)
            state, reward, terminated, truncated, info = env.step(np.clip(act, 0, 512).astype(np.float32))
            history = [history[-1], np.asarray(state, dtype=np.float32)]
            best_reward = max(best_reward, float(reward))
            if terminated:
                success = True
            if terminated or truncated:
                break
        rewards.append(best_reward)
        successes.append(int(success))
        print(f"episode {ep+1}/{args.episodes}: success={success}, max_reward={best_reward:.3f}", flush=True)
    env.close()
    print(f"RESULT mode={args.mode}: success_rate={np.mean(successes):.3f}, max_reward_mean={np.mean(rewards):.3f}, n={args.episodes}")
    print("WARNING: these are local results of a light-weight implementation, NOT paper benchmark scores.")


def smoke(args):
    print("Synthetic selftest (no robot demonstrations and no gym required):")
    dev = device_of(args.device)
    states, actions, ends = load_arrays(synthetic=True)
    ds = WindowDataset(states, actions, ends, "train")
    obs, act = next(iter(DataLoader(ds, batch_size=4, num_workers=0)))
    policy = Denoiser(hidden=64).to(dev)
    loss = policy.loss(obs.to(dev), act.to(dev))
    loss.backward()
    sample = policy.sample(obs.to(dev), 4)
    tr = TransitionDataset(states, actions, ends, "train")
    s, a, sp = next(iter(DataLoader(tr, batch_size=4, num_workers=0)))
    wm = WorldMLP(hidden=32).to(dev)
    wloss = F.mse_loss(wm(s.to(dev), a.to(dev)), sp.to(dev))
    wloss.backward()
    assert sample.shape == (4, 8, 2)
    print("PASS: policy forward/backward and sample", tuple(sample.shape), "loss", float(loss))
    print("PASS: world forward/backward", "loss", float(wloss))


def main():
    p = argparse.ArgumentParser(description="Windows 8GB PushT world-model + diffusion policy learning project")
    sub = p.add_subparsers(dest="command", required=True)
    for c in ["check", "download", "inspect", "smoke", "train-policy", "train-world", "eval"]:
        q = sub.add_parser(c)
        if c in ("smoke", "train-policy", "train-world", "eval"):
            q.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
            q.add_argument("--seed", type=int, default=42)
        if c in ("train-policy", "train-world"):
            q.add_argument("--steps", type=int, default=100)
            q.add_argument("--batch-size", type=int, default=16)
            q.add_argument("--hidden", type=int, default=256 if c == "train-policy" else 128)
            q.add_argument("--lr", type=float, default=1e-4)
            q.add_argument("--log-every", type=int, default=50)
            q.add_argument("--synthetic", action="store_true", help="Only use for debugging; NOT real training")
            if c == "train-policy":
                q.add_argument("--horizon", type=int, default=8)
        if c == "eval":
            q.add_argument("--mode", choices=["policy", "sample4", "world"], default="policy")
            q.add_argument("--episodes", type=int, default=5)
            q.add_argument("--max-steps", type=int, default=200)
            q.add_argument("--candidates", type=int, default=4)
            q.add_argument("--sample-steps", type=int, default=8)
    args = p.parse_args()
    if hasattr(args, "seed"):
        seed_all(args.seed)
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
