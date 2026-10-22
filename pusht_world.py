"""History-conditioned dynamics and conservative candidate ranking for PushT.

All tensors use the same normalized coordinates as wm_pusht8g. No simulator
state, reward, or future observation is consulted during candidate selection.
"""
from __future__ import annotations

import math
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def state_difference(pred, target):
    delta = pred - target
    angle = delta[..., 4:5] * math.pi
    return torch.cat((delta[..., :4], torch.atan2(angle.sin(), angle.cos()) / math.pi), -1)


def periodic_state(state):
    angle = (state[..., 4:5] + 1) * math.pi
    relative = state[..., :2] - state[..., 2:4]
    local = torch.cat((relative[..., :1] * angle.cos() + relative[..., 1:] * angle.sin(),
                       -relative[..., :1] * angle.sin() + relative[..., 1:] * angle.cos()), -1)
    return torch.cat((state[..., :4], angle.sin(), angle.cos(), local), -1)


class RolloutDataset(Dataset):
    """Whole-episode split; targets never repeat the terminal state or cross episodes."""
    def __init__(self, states, actions, ends, split='train', history_len=4, horizon=4, seed=42):
        if history_len < 2 or horizon < 1 or len(ends) < 2:
            raise ValueError('World training needs history >= 2, horizon >= 1 and >= 2 episodes')
        scale = np.array([256, 256, 256, 256, math.pi], np.float32)
        self.states = torch.from_numpy((states / scale - 1).astype(np.float32))
        self.actions = torch.from_numpy((actions / 256 - 1).astype(np.float32))
        self.history_len, self.horizon = history_len, horizon
        order = np.random.default_rng(seed).permutation(len(ends))
        self.train_episodes = set(order[:min(len(ends)-1, max(1, int(len(ends)*.9)))].tolist())
        self.indices = []
        for ep, (start, end) in enumerate(zip(np.r_[0, ends[:-1]], ends)):
            if (ep in self.train_episodes) == (split == 'train'):
                self.indices.extend((t, int(start), int(end), ep)
                                    for t in range(int(start), int(end)-horizon))
        if not self.indices:
            raise ValueError(f'No complete {horizon}-step windows in {split} episodes')
        self.episodes = sorted({r[3] for r in self.indices})

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        t, start, _, _ = self.indices[index]
        times = np.arange(t-self.history_len+1, t+1)
        history = self.states[np.maximum(times, start)]
        # Before an episode starts, use a stationary setpoint, not a future expert action.
        previous = torch.stack([self.actions[i] if i >= start else self.states[start, :2]
                                for i in times[:-1]])
        return history, previous, self.actions[t:t+self.horizon], self.states[t+1:t+self.horizon+1]


class HistoryDynamics(nn.Module):
    def __init__(self, hidden=128, history_len=4):
        super().__init__()
        self.history_len = history_len
        features = 8*history_len + 7*(history_len-1) + 2
        self.net = nn.Sequential(nn.Linear(features, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 5))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('delta_scale', torch.tensor([.1, .1, .04, .04, .04]))

    def forward(self, history, previous_actions, action):
        motion = state_difference(history[:, 1:], history[:, :-1])
        x = torch.cat((periodic_state(history).flatten(1), motion.flatten(1),
                       previous_actions.flatten(1), action), -1)
        result = history[:, -1] + self.net(x) * self.delta_scale
        # Canonical angle, without clipping positions and hiding divergent rollouts.
        angle = torch.remainder(result[:, 4:5] + 1, 2) - 1
        return torch.cat((result[:, :4], angle), -1)

    def rollout(self, history, previous_actions, actions):
        predictions = []
        for action in actions.unbind(1):
            nxt = self(history, previous_actions, action)
            predictions.append(nxt)
            history = torch.cat((history[:, 1:], nxt[:, None]), 1)
            previous_actions = torch.cat((previous_actions[:, 1:], action[:, None]), 1)
        return torch.stack(predictions, 1)


class WorldEnsemble(nn.Module):
    version = 2

    def __init__(self, hidden=128, history_len=4, members=3, horizon=4):
        super().__init__()
        if members < 2:
            raise ValueError('At least two world members are needed for disagreement estimates')
        self.members = nn.ModuleList([HistoryDynamics(hidden, history_len) for _ in range(members)])
        self.hidden, self.history_len, self.horizon = hidden, history_len, horizon
        self.register_buffer('error_scale', torch.tensor([.05, .05, .02, .02, .03]))
        self.register_buffer('uncertainty_limit', torch.tensor(float('nan')))
        self.register_buffer('support_limit', torch.tensor(float('nan')))
        self.register_buffer('support_bank', torch.empty(0, 10))

    def rollout(self, history, previous_actions, actions):
        return torch.stack([m.rollout(history, previous_actions, actions) for m in self.members])

    def disagreement(self, predictions):
        # Pairwise circular difference avoids +pi/-pi variance artifacts.
        pairs = [((state_difference(predictions[i], predictions[j]) / self.error_scale)**2).mean(-1)
                 for i in range(len(self.members)) for j in range(i)]
        return torch.stack(pairs).mean(0).sqrt()

    @staticmethod
    def support_features(states, actions):
        return torch.cat((periodic_state(states), actions), -1)

    def support_distance(self, states, actions):
        query = self.support_features(states, actions)
        shape = query.shape[:-1]
        if not len(self.support_bank):
            return torch.full(shape, float("inf"), device=states.device)
        query = query.reshape(-1, 10)
        distances = []
        for part in query.split(256):
            distances.append(torch.cdist(part, self.support_bank).amin(-1) / math.sqrt(10))
        return torch.cat(distances).reshape(shape)


def rollout_loss(predictions, targets, scale):
    return (state_difference(predictions, targets) / scale).square().mean()


@torch.no_grad()
def validate_world(model, loader, device, calibrate=False):
    """Entire held-out split, autoregressive targets; sample-weighted metrics."""
    model.eval()
    n, total = 0, 0.
    h = model.horizon
    agent = torch.zeros(h, device=device)
    block, angle = agent.clone(), agent.clone()
    disagreement, actual_error, support = [], [], []
    for history, previous, actions, targets in loader:
        history, previous, actions, targets = [x.to(device) for x in (history, previous, actions, targets)]
        predictions = model.rollout(history, previous, actions)
        error = state_difference(predictions, targets.unsqueeze(0))
        total += (error/model.error_scale).square().mean().item()*len(history)
        agent += error[..., :2].norm(dim=-1).mean(0).sum(0)*256
        block += error[..., 2:4].norm(dim=-1).mean(0).sum(0)*256
        angle += error[..., 4].abs().mean(0).sum(0)*180
        n += len(history)
        if calibrate:
            disagreement.append(model.disagreement(predictions).amax(-1).cpu())
            actual_error.append((error/model.error_scale).square().mean((0, 2, 3)).sqrt().cpu())
            # Score the states actually queried during autoregressive planning.
            rollout_states = torch.cat((history[:, -1][None, :, None].expand(len(model.members), -1, 1, -1),
                                        predictions[:, :, :-1]), 2)
            action_batch = actions[None].expand(len(model.members), -1, -1, -1)
            support.append(model.support_distance(rollout_states, action_batch).amax((0, 2)).cpu())
    result = {'loss': total/n, 'windows': n, 'agent_error_px_by_step': (agent/n).tolist(),
              'block_error_px_by_step': (block/n).tolist(), 'angle_error_deg_by_step': (angle/n).tolist()}
    if calibrate:
        uncertainty, error = torch.cat(disagreement), torch.cat(actual_error)
        support = torch.cat(support)
        # Held-out demonstration quantiles are a conservative heuristic, NOT calibrated probabilities.
        model.uncertainty_limit.copy_(uncertainty.quantile(.95).clamp_min(1e-4))
        model.support_limit.copy_(support.quantile(.95).clamp_min(1e-4))
        corr = None
        if uncertainty.std() > 0 and error.std() > 0:
            corr = float(torch.corrcoef(torch.stack((uncertainty, error)))[0, 1])
        result.update(uncertainty_limit=float(model.uncertainty_limit), support_limit=float(model.support_limit),
                      uncertainty_error_correlation=corr,
                      calibration='95th percentile of full held-out demonstration windows; heuristic only')
    return result


def atomic_save(checkpoint, path):
    temp = path.with_suffix('.tmp')
    torch.save(checkpoint, temp)
    temp.replace(path)


def train_ensemble(args, states, actions, ends, out_dir, device):
    train = RolloutDataset(states, actions, ends, history_len=args.world_history,
                           horizon=args.world_horizon, seed=args.seed)
    val = RolloutDataset(states, actions, ends, 'val', args.world_history, args.world_horizon, args.seed)
    model = WorldEnsemble(args.hidden, args.world_history, args.world_members, args.world_horizon).to(device)
    # Independent episode bootstraps plus independent minibatch draws for each member.
    rng = np.random.default_rng(args.seed+971)
    episode_rows = {ep: np.array([i for i, row in enumerate(train.indices) if row[3] == ep])
                    for ep in train.episodes}
    pools = [np.concatenate([episode_rows[int(ep)] for ep in
                             rng.choice(train.episodes, len(train.episodes), replace=True)])
             for _ in model.members]
    loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.steps, eta_min=args.lr*.1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir/'world_metrics.jsonl'
    log_path.write_text('', encoding='utf-8')
    best = float('inf')
    # Materialize small low-dimensional windows once, avoiding per-step Python dataset overhead.
    tensors = [torch.stack(items).to(device) for items in zip(*(train[i] for i in range(len(train))))]
    bank_rows = rng.choice(len(train), min(2048, len(train)), replace=False)
    model.support_bank = model.support_features(tensors[0][bank_rows, -1], tensors[2][bank_rows, 0]).detach()
    metadata = {'world_version': 2, 'hidden': args.hidden, 'history_len': args.world_history,
                'members': args.world_members, 'horizon': args.world_horizon, 'seed': args.seed,
                'synthetic': args.synthetic, 'train_episodes': train.episodes, 'val_episodes': val.episodes,
                'train_windows': len(train), 'val_windows': len(val), 'batch_size': args.batch_size,
                'learning_rate': args.lr, 'total_updates': args.steps}
    print(f'World train={len(train)} val={len(val)} windows, members={len(model.members)}, '
          f'history={model.history_len}, rollout={model.horizon}', flush=True)
    for step in range(1, args.steps+1):
        model.train()
        optim.zero_grad(set_to_none=True)
        losses = []
        for member, pool in zip(model.members, pools):
            idx = torch.as_tensor(rng.choice(pool, args.batch_size, replace=True), device=device)
            history, previous, act, target = [x[idx] for x in tensors]
            loss = rollout_loss(member.rollout(history, previous, act), target, model.error_scale)
            loss.backward()
            losses.append(float(loss.detach()))
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optim.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            metrics = validate_world(model, loader, device)
            record = {'step': step, 'train_loss': float(np.mean(losses)), 'validation': metrics}
            with log_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record)+'\n')
            print(f'world step={step} train={np.mean(losses):.5f} val={metrics["loss"]:.5f} '
                  f'block@{model.horizon}={metrics["block_error_px_by_step"][-1]:.2f}px '
                  f'angle={metrics["angle_error_deg_by_step"][-1]:.2f}deg', flush=True)
            checkpoint = dict(metadata, model=model.state_dict(), steps=step, validation=metrics)
            atomic_save(checkpoint, out_dir/'world_last.pt')
            if metrics['loss'] < best:
                best = metrics['loss']
                atomic_save(checkpoint, out_dir/'world.pt')
    selected = torch.load(out_dir/'world.pt', map_location=device, weights_only=True)
    model.load_state_dict(selected['model'])
    metrics = validate_world(model, loader, device, calibrate=True)
    selected.update(model=model.state_dict(), validation=metrics)
    atomic_save(selected, out_dir/'world.pt')
    # Also calibrate the final checkpoint so explicitly loading it remains safe and useful.
    last = torch.load(out_dir/'world_last.pt', map_location=device, weights_only=True)
    model.load_state_dict(last['model'])
    last.update(validation=validate_world(model, loader, device, calibrate=True), model=model.state_dict())
    atomic_save(last, out_dir/'world_last.pt')
    summary = {k: v for k, v in selected.items() if k != 'model'}
    (out_dir/'world_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Saved selected world at update {selected["steps"]}: {out_dir / "world.pt"}', flush=True)


def load_ensemble(checkpoint, device):
    model = WorldEnsemble(checkpoint['hidden'], checkpoint['history_len'],
                          checkpoint['members'], checkpoint['horizon']).to(device)
    model.support_bank = torch.empty_like(checkpoint['model']['support_bank'], device=device)
    model.load_state_dict(checkpoint['model'])
    model.checkpoint_metadata = {k: v for k, v in checkpoint.items() if k != 'model'}
    if checkpoint.get('synthetic'):
        print('WARNING: synthetic world checkpoint; not suitable for task performance claims.')
    return model.eval()


TEE_VERTICES = np.array([[-60, 0], [60, 0], [60, 30], [15, 30],
                         [15, 120], [-15, 120], [-15, 30], [-60, 30]], dtype=np.float64)


def polygon_at(pose):
    from shapely.geometry import Polygon
    c, s = np.cos(pose[2]), np.sin(pose[2])
    vertices = TEE_VERTICES @ np.array([[c, s], [-s, c]]) + pose[:2]
    return Polygon(vertices)


def coverage_from_states(states, goal_pose=(256., 256., math.pi/4)):
    """Exact geometric area ratio used by stock gym-pusht, not rounded reward."""
    states = np.asarray(states)
    goal = polygon_at(np.asarray(goal_pose))
    values = [polygon_at(s[2:5]).intersection(goal).area / goal.area for s in states.reshape(-1, 5)]
    return np.asarray(values).reshape(states.shape[:-1])


def history_tensors(history, executed_actions, length, device):
    # Input history has real observations only; actions correspond to transitions between them.
    history = np.asarray(history[-length:], dtype=np.float32)
    missing = length-len(history)
    previous = np.asarray(executed_actions[-(len(history)-1):] if len(history)>1 else [], dtype=np.float32).reshape(-1, 2)
    if len(previous) != len(history)-1:
        raise ValueError('World history requires one executed action per observed transition')
    if missing:
        previous = np.concatenate((np.repeat(history[:1, :2], missing, axis=0), previous))
        history = np.concatenate((np.repeat(history[:1], missing, axis=0), history))
    scale = np.array([256, 256, 256, 256, math.pi], dtype=np.float32)
    return (torch.as_tensor(history/scale-1, device=device)[None],
            torch.as_tensor(previous/256-1, device=device)[None])


@torch.no_grad()
def rank_candidates(world, history, previous, candidates, plan_steps, last_action=None,
                    uncertainty_weight=.15, switch_margin=.02, goal_pose=(256., 256., math.pi/4)):
    """Conservative reranking: candidate zero is an unchanged policy proposal.

    Disagreement AND distance from training support gate overrides. These are
    diagnostics/heuristics, not probabilities or safety guarantees.
    """
    if not 1 <= plan_steps <= min(world.horizon, candidates.shape[1]):
        raise ValueError('Planning horizon must not exceed trained horizon or available actions')
    count = len(candidates)
    history = history.expand(count, -1, -1)
    previous = previous.expand(count, -1, -1)
    actions = candidates[:, :plan_steps]
    predictions = world.rollout(history, previous, actions)
    uncertainty = world.disagreement(predictions).amax(-1)
    query_states = torch.cat((history[:, -1][None, :, None].expand(len(world.members), -1, 1, -1),
                              predictions[:, :, :-1]), 2)
    support = world.support_distance(query_states, actions[None].expand(len(world.members), -1, -1, -1)).amax((0, 2))
    # Invalid geometry and nonfinite outputs can never win a ranking.
    finite = torch.isfinite(predictions).all((0, 2, 3))
    finite &= (predictions[..., :4].abs() < 3).all((0, 2, 3))
    scores = torch.full((count,), float('inf'), device=candidates.device)
    if finite.any():
        valid = predictions[:, finite]
        raw = (valid.cpu().numpy()+1) * np.array([256, 256, 256, 256, math.pi])
        coverage = torch.as_tensor(coverage_from_states(raw, goal_pose), device=candidates.device, dtype=candidates.dtype)
        goal = candidates.new_tensor([goal_pose[0]/256-1, goal_pose[1]/256-1])
        position = (valid[..., 2:4]-goal).square().sum(-1)
        theta = (valid[..., 4]+1)*math.pi - goal_pose[2]
        angle = torch.atan2(theta.sin(), theta.cos()).square()
        trajectory_cost = (1-coverage) + .1*position + .05*angle
        cost = .75*trajectory_cost[..., -1] + .25*trajectory_cost.mean(-1)
        smooth_actions = actions[finite]
        if last_action is not None:
            prior = candidates.new_tensor(last_action)/256-1
            smooth_actions = torch.cat((prior[None, None].expand(len(smooth_actions), 1, 2), smooth_actions), 1)
        smoothness = (smooth_actions[:, 1:]-smooth_actions[:, :-1]).square().mean((1, 2)) if smooth_actions.shape[1]>1 else cost.new_zeros(cost.shape[1])
        scores[finite] = cost.mean(0) + .02*smoothness + uncertainty_weight*uncertainty[finite]
    reliable = finite & torch.isfinite(uncertainty) & torch.isfinite(support)
    reliable &= (uncertainty <= world.uncertainty_limit) & (support <= world.support_limit)
    alternatives = scores.clone()
    alternatives[~reliable] = float('inf')
    proposed = int(alternatives.argmin())
    selected, reason = 0, 'no_reliable_improvement'
    # If the original proposal cannot be evaluated reliably, do not replace it on a speculative comparison.
    if bool(reliable[0]) and bool(reliable[proposed]) and proposed != 0 and scores[proposed]+switch_margin < scores[0]:
        selected, reason = proposed, 'world_improvement'
    elif not bool(reliable.any()):
        reason = 'all_unreliable'
    elif not bool(reliable[0]):
        reason = 'baseline_unreliable'
    info = {'selected': selected, 'reason': reason, 'reliable_candidates': int(reliable.sum()),
            'uncertainty': float(uncertainty[selected]) if torch.isfinite(uncertainty[selected]) else None,
            'support_distance': float(support[selected]) if torch.isfinite(support[selected]) else None,
            'plan_steps': plan_steps}
    return selected, info
