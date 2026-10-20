"""Regression checks for training/deployment semantics (no benchmark claims)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import wm_pusht8g as wm


class PushTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_terminal_noise_and_schedule(self):
        model = wm.Denoiser(hidden=32)
        self.assertLess(model.alpha_bars[-1].item(), 1e-5)
        self.assertTrue(torch.all(model.alpha_bars[1:] < model.alpha_bars[:-1]))
        old = wm.Denoiser(hidden=32, diffusion_steps=32, architecture="legacy", schedule="linear")
        self.assertGreater(old.alpha_bars[-1].item(), 0.7)

    def test_oracle_sampler_recovers_actions(self):
        for prediction_type in ("epsilon", "v_prediction"):
            model = wm.Denoiser(hidden=32, horizon=8, prediction_type=prediction_type)
            target = torch.linspace(-0.8, 0.8, 16).reshape(1, 8, 2)
            def oracle(obs, noisy, t):
                ab = model.alpha_bars[t].reshape(-1, 1, 1)
                eps = (noisy - ab.sqrt() * target) / (1 - ab).sqrt()
                return eps if prediction_type == "epsilon" else ab.sqrt() * eps - (1 - ab).sqrt() * target
            with patch.object(model, "forward", side_effect=oracle):
                result = model.sample(torch.zeros(1, 2, 5), 32)
            torch.testing.assert_close(result, target, atol=1e-5, rtol=1e-5)

    def test_angle_wrap_does_not_change_prediction(self):
        model = wm.Denoiser(hidden=32, horizon=8)
        obs = torch.randn(2, 2, 5)
        wrapped = obs.clone()
        wrapped[..., 4] += 2  # normalized equivalent of 2*pi
        action, t = torch.randn(2, 8, 2), torch.tensor([10, 90])
        torch.testing.assert_close(model(obs, action, t), model(wrapped, action, t), atol=2e-6, rtol=2e-5)

    def test_geometry_features_match_environment_shape(self):
        import gymnasium as gym
        import gym_pusht  # noqa: F401
        env = gym.make("gym_pusht/PushT-v0")
        try:
            for seed in (1, 42, 10000):
                obs, _ = env.reset(seed=seed)
                feature = wm.policy_features(torch.tensor(wm.norm_s(obs), dtype=torch.float32))
                predicted = ((feature[2:18].reshape(8, 2) + 1) * 256).numpy()
                vertices = np.unique(np.array([env.unwrapped.block.local_to_world(v)
                    for shape in env.unwrapped.block.shapes for v in shape.get_vertices()]), axis=0)
                self.assertEqual(len(vertices), 8)
                for point in predicted:
                    self.assertLess(np.linalg.norm(vertices - point, axis=1).min(), 0.001)
        finally:
            env.close()

    def test_dataset_alignment_boundaries_and_split(self):
        states, actions, ends = wm.load_arrays(True)
        train = wm.WindowDataset(states, actions, ends, horizon=16, seed=9)
        val = wm.WindowDataset(states, actions, ends, "val", horizon=16, seed=9)
        self.assertFalse({s for _, s, _ in train.indices} & {s for _, s, _ in val.indices})
        for dataset in (train, val):
            for i, (t, start, end) in enumerate(dataset.indices):
                obs, act = dataset[i]
                np.testing.assert_allclose(obs[-1], wm.norm_s(states[t]))
                np.testing.assert_allclose(obs[0], wm.norm_s(states[max(start, t - 1)]))
                np.testing.assert_allclose(act[0], wm.norm_a(actions[t]))
                np.testing.assert_allclose(act[-1], wm.norm_a(actions[min(t + 15, end - 1)]))

    def test_chunk_executes_once_and_resets(self):
        policy = wm.Denoiser(hidden=32, horizon=8)
        controller = wm.ChunkController(policy, action_steps=4)
        chunk = np.arange(16, dtype=np.float32).reshape(8, 2)
        with patch.object(wm, "select_chunk", return_value=chunk) as sampler:
            for i in range(4):
                np.testing.assert_equal(controller.act(np.full(5, i)), chunk[i])
            self.assertEqual(sampler.call_count, 1)
            np.testing.assert_equal(controller.history[-1], np.full(5, 3))
            np.testing.assert_equal(controller.act(np.full(5, 4)), chunk[0])
            self.assertEqual(sampler.call_count, 2)
            controller.reset()
            np.testing.assert_equal(controller.act(np.full(5, 8)), chunk[0])
            self.assertEqual(sampler.call_count, 3)

    def test_legacy_checkpoint_keeps_original_schedule(self):
        original = wm.Denoiser(hidden=32, horizon=8, diffusion_steps=32,
                               architecture="legacy", schedule="linear")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            torch.save({"model": original.state_dict(), "hidden": 32, "horizon": 8}, path)
            restored = wm.load_policy(torch.device("cpu"), path)
        torch.testing.assert_close(restored.alpha_bars, original.alpha_bars)
        self.assertEqual(restored.architecture, "legacy")

    def test_invalid_sampling_settings(self):
        model = wm.Denoiser(hidden=32)
        for steps in (0, 1, 101):
            with self.assertRaises(ValueError):
                model.sample(torch.zeros(1, 2, 5), steps)
        with self.assertRaises(ValueError):
            wm.ChunkController(model, action_steps=17)

    def test_temporal_model_horizons_and_backward(self):
        for horizon in (3, 8, 16):
            model = wm.Denoiser(hidden=16, horizon=horizon, architecture="unet")
            obs, action = torch.randn(2, 2, 5), torch.rand(2, horizon, 2) * 2 - 1
            loss = model.loss(obs, action)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            sample = model.sample(obs, 4)
            self.assertEqual(sample.shape, action.shape)
            self.assertTrue(torch.isfinite(sample).all())


if __name__ == "__main__":
    unittest.main()
