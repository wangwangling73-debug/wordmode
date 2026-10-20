"""PushT visualization for the companion wm_pusht8g.py starter.

Run live (Pygame window):
    python visualize_pusht.py --mode random
    python visualize_pusht.py --mode policy --device cuda

Save a GIF (without displaying a window):
    python visualize_pusht.py --mode random --record outputs/random.gif
    python visualize_pusht.py --mode policy --device cuda --record outputs/policy.gif
    python visualize_pusht.py --mode world --device cuda --record outputs/world.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wm_pusht8g_v2 import device_of, load_policy, load_world, seed_all, select_chunk


def run(args):
    import gymnasium as gym
    import gym_pusht  # noqa: F401, registers gym_pusht/PushT-v0

    recording = bool(args.record)
    # gym-pusht only supports one render mode per environment instance.
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="state",
        render_mode="rgb_array" if recording else "human",
        visualization_width=512,
        visualization_height=512,
    )
    policy = world = None
    if args.mode != "random":
        dev = device_of(args.device)
        policy = load_policy(dev)  # requires runs/policy.pt
        if args.mode == "world":
            world = load_world(dev)  # requires runs/world.pt

    seed_all(args.seed)
    frames = []
    try:
        obs, info = env.reset(seed=args.seed)
        history = [np.asarray(obs, dtype=np.float32)] * 2
        if recording:
            frames.append(np.asarray(env.render(), dtype=np.uint8).copy())
        else:
            env.render()
        best_reward = 0.0
        success = False
        steps = 0
        pending = []
        for t in range(args.steps):
            if args.mode == "random":
                action = env.action_space.sample()
            else:
                if not pending:
                    chunk = select_chunk(policy, world, history, args.mode,
                                         args.candidates, args.sample_steps)
                    pending = list(np.clip(chunk[:args.action_steps], 0, 512).astype(np.float32))
                action = pending.pop(0)
            obs, reward, terminated, truncated, info = env.step(action)
            history = [history[-1], np.asarray(obs, dtype=np.float32)]
            best_reward = max(best_reward, float(reward))
            success = success or bool(terminated)
            steps = t + 1
            if recording and t % args.stride == 0:
                frames.append(np.asarray(env.render(), dtype=np.uint8).copy())
            elif not recording:
                env.render()
            if steps % 25 == 0 or terminated or truncated:
                print(f"step={steps:3d}, reward={reward:.3f}, max_reward={best_reward:.3f}, success={success}", flush=True)
            if terminated or truncated:
                break
        print(f"Finished: mode={args.mode}, steps={steps}, success={success}, max_reward={best_reward:.3f}")
        if recording:
            import imageio.v2 as imageio
            out = Path(args.record)
            out.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(out, frames, duration=args.stride / args.fps, loop=0)
            print(f"Saved animation: {out.resolve()} ({len(frames)} frames)")
        else:
            print("The simulation window will close when the script finishes.")
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description="Visualize PushT live or record a GIF, Windows compatible")
    p.add_argument("--mode", choices=["random", "policy", "sample4", "world"], default="random")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=10, help="Playback FPS for saved GIF")
    p.add_argument("--stride", type=int, default=2, help="Save every Nth step to reduce GIF size")
    p.add_argument("--action-steps", type=int, default=4, help="Execute up to N predicted actions before replanning")
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--sample-steps", type=int, default=16)
    p.add_argument("--record", type=str, default=None, metavar="FILE.gif",
                   help="Save GIF instead of opening live simulation window")
    args = p.parse_args()
    if args.fps < 1 or args.stride < 1 or args.steps < 1 or args.action_steps < 1:
        p.error("--fps, --stride, --steps and --action-steps must be positive integers")
    if args.record and not args.record.lower().endswith(".gif"):
        p.error("--record supports .gif output; please use a filename ending in .gif")
    try:
        run(args)
    except FileNotFoundError as exc:
        p.exit(2, f"Model checkpoint not found: {exc}\nRun training first (see README).\n")


if __name__ == "__main__":
    main()
