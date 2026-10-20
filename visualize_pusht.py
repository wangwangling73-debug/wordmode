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
import torch

from wm_pusht8g import ChunkController, device_of, load_policy, load_world, seed_all


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
        max_episode_steps=args.steps,
    )
    policy = world = None
    if args.mode != "random":
        dev = device_of(args.device)
        policy = load_policy(dev, args.checkpoint)
        if args.mode == "world":
            world = load_world(dev, args.world_checkpoint)
        controller = ChunkController(policy, world, args.mode, args.candidates,
                                     args.sample_steps, args.action_steps)

    seed_all(args.seed)
    env.action_space.seed(args.seed)
    frames = []
    try:
        obs, info = env.reset(seed=args.seed)
        if recording:
            frames.append(np.asarray(env.render(), dtype=np.uint8).copy())
        else:
            env.render()
        best_reward = 0.0
        success = False
        steps = 0
        for t in range(args.steps):
            if args.mode == "random":
                action = env.action_space.sample()
            else:
                action = controller.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            best_reward = max(best_reward, float(reward))
            success = success or bool(terminated)
            steps = t + 1
            if recording and (steps % args.stride == 0 or terminated or truncated or steps == args.steps):
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
            imageio.mimsave(out, frames, duration=1000 * args.stride / args.fps, loop=0)
            print(f"Saved animation: {out.resolve()} ({len(frames)} frames)")
        else:
            print("The simulation window will close when the script finishes.")
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description="Visualize PushT live or record a GIF, Windows compatible")
    p.add_argument("--mode", choices=["random", "policy", "sample4", "world"], default="random")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=10, help="Playback FPS for saved GIF")
    p.add_argument("--stride", type=int, default=2, help="Save every Nth step to reduce GIF size")
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--sample-steps", type=int, default=32)
    p.add_argument("--action-steps", type=int, default=4)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--world-checkpoint", type=Path)
    p.add_argument("--record", type=str, default=None, metavar="FILE.gif",
                   help="Save GIF instead of opening live simulation window")
    args = p.parse_args()
    if min(args.fps, args.stride, args.steps, args.action_steps, args.candidates, args.cpu_threads) < 1 or args.sample_steps < 2:
        p.error("Counts must be positive; --sample-steps must be at least 2")
    if args.record and not args.record.lower().endswith(".gif"):
        p.error("--record supports .gif output; please use a filename ending in .gif")
    torch.set_num_threads(args.cpu_threads)
    try:
        run(args)
    except FileNotFoundError as exc:
        p.exit(2, f"Model checkpoint not found: {exc}\nRun training first (see README).\n")


if __name__ == "__main__":
    main()
