"""World-model regression tests: temporal alignment, geometry, and abstention."""
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import wm_pusht8g as wm
from pusht_world import (RolloutDataset, HistoryDynamics, WorldEnsemble, state_difference,
                         coverage_from_states, history_tensors, rank_candidates, validate_world,
                         load_ensemble)


class WorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_episode_split_history_and_future_alignment(self):
        states, actions, ends = wm.load_arrays(True)
        train = RolloutDataset(states, actions, ends, history_len=4, horizon=4)
        val = RolloutDataset(states, actions, ends, 'val', 4, 4)
        self.assertFalse(set(train.episodes) & set(val.episodes))
        for dataset in (train, val):
            for i, (t, start, end, _) in enumerate(dataset.indices):
                history, previous, future, target = dataset[i]
                self.assertLess(t+4, end)
                np.testing.assert_allclose(history[-1], wm.norm_s(states[t]), atol=1e-6)
                np.testing.assert_allclose(future, wm.norm_a(actions[t:t+4]), atol=1e-6)
                np.testing.assert_allclose(target, wm.norm_s(states[t+1:t+5]), atol=1e-6)
                for j, old in enumerate(range(t-3, t)):
                    expected = actions[old] if old >= start else states[start, :2]
                    np.testing.assert_allclose(previous[j], wm.norm_a(expected), atol=1e-6)

    def test_periodic_motion_and_prediction(self):
        model = HistoryDynamics(16, 4)
        torch.nn.init.normal_(model.net[-1].weight, std=.1)
        history, previous, actions = torch.randn(2, 4, 5), torch.rand(2, 3, 2), torch.rand(2, 4, 2)
        wrapped = history.clone()
        wrapped[..., 4] += 2
        predicted = model.rollout(history, previous, actions)
        equivalent = model.rollout(wrapped, previous, actions)
        torch.testing.assert_close(state_difference(predicted, equivalent), torch.zeros_like(predicted), atol=2e-6, rtol=0)
        predicted.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_rollout_feeds_predictions_and_executed_actions(self):
        model = HistoryDynamics(16, 3)
        observations = []
        def forward(history, previous, action):
            observations.append((history.clone(), previous.clone(), action.clone()))
            return history[:, -1] + 1
        actions = torch.arange(8.).reshape(1, 4, 2)
        with patch.object(model, 'forward', side_effect=forward):
            result = model.rollout(torch.zeros(1, 3, 5), torch.zeros(1, 2, 2), actions)
        torch.testing.assert_close(result[0, :, 0], torch.arange(1., 5.))
        torch.testing.assert_close(observations[2][1][0], actions[0, :2])

    def test_exact_coverage_matches_simulator(self):
        import gymnasium as gym
        import gym_pusht
        with gym.make('gym_pusht/PushT-v0', obs_type='state') as env:
            for seed in (1, 42, 101):
                state, _ = env.reset(seed=seed)
                for _ in range(3):
                    state, _, _, _, info = env.step(np.array([256., 256.]))
                    self.assertAlmostEqual(float(coverage_from_states(state)), info['coverage'], places=6)
        self.assertAlmostEqual(float(coverage_from_states(np.array([0, 0, 256, 256, math.pi/4]))), 1.)

    def test_history_only_records_actions_with_observed_transitions(self):
        policy, world = wm.Denoiser(hidden=16, horizon=8), WorldEnsemble(16)
        controller = wm.ChunkController(policy, world, 'world')
        chunk = np.arange(16, dtype=np.float32).reshape(8, 2)+100
        with patch.object(wm, 'select_chunk', return_value=chunk):
            controller.act(np.zeros(5))
            self.assertEqual(len(controller.executed_actions), 0)
            controller.act(np.ones(5))
            np.testing.assert_equal(controller.executed_actions, chunk[:1])
            history, previous = history_tensors(controller.history, controller.executed_actions, 4, 'cpu')
            np.testing.assert_allclose(previous[0, -1], wm.norm_a(chunk[0]))
            np.testing.assert_equal(previous[0, :2], -np.ones((2, 2)))
            controller.reset()
            self.assertEqual(controller.executed_actions, [])
            self.assertEqual(controller.decisions, [])
            self.assertIsNone(controller.last_action)

    def test_disagreement_handles_equivalent_angles(self):
        world = WorldEnsemble(16, members=2)
        prediction = torch.zeros(2, 3, 4, 5)
        prediction[1, ..., 4] = 2
        self.assertLess(float(world.disagreement(prediction).max()), 1e-5)

    def test_ranking_and_unreliable_fallback(self):
        world = WorldEnsemble(16, members=2)
        world.uncertainty_limit.fill_(1)
        world.support_limit.fill_(1)
        history, previous, actions = torch.zeros(1, 4, 5), torch.zeros(1, 3, 2), torch.zeros(2, 4, 2)
        raw = torch.tensor([200., 200., 100., 100., 0.])
        bad = torch.tensor(wm.norm_s(raw.numpy()))
        good = torch.tensor(wm.norm_s(np.array([200., 200., 256., 256., math.pi/4], dtype=np.float32)))
        predictions = torch.stack((bad, good))[None, :, None].expand(2, 2, 4, 5).clone()
        with patch.object(world, 'rollout', return_value=predictions), patch.object(world, 'support_distance', return_value=torch.zeros(2, 2, 4)):
            self.assertEqual(rank_candidates(world, history, previous, actions, 4)[0], 1)
            with patch.object(world, 'disagreement', return_value=torch.full((2, 4), 2.)):
                selected, info = rank_candidates(world, history, previous, actions, 4)
                self.assertEqual(selected, 0)
                self.assertEqual(info['reason'], 'all_unreliable')
            predictions[:, 1] = float('nan')
            self.assertEqual(rank_candidates(world, history, previous, actions, 4)[0], 0)
        with patch.object(world, 'rollout', return_value=torch.zeros(2, 2, 4, 5)):
            self.assertEqual(rank_candidates(world, history, previous, actions, 4)[1]['reason'], 'all_unreliable')

    def test_candidate_zero_and_rng_match_baseline(self):
        policy = wm.Denoiser(hidden=16, horizon=8)
        world = WorldEnsemble(16)
        state = np.array([256., 256., 200., 200., .1], dtype=np.float32)
        torch.manual_seed(123)
        baseline = wm.select_chunk(policy, None, [state], 'policy', 4, 4)
        next_noise = torch.rand(5)
        torch.manual_seed(123)
        with patch.object(wm, 'rank_candidates', return_value=(0, {})):
            fallback = wm.select_chunk(policy, world, [state], 'world', 4, 4)
        np.testing.assert_equal(fallback, baseline)
        torch.testing.assert_close(torch.rand(5), next_noise)
        torch.manual_seed(123)
        np.testing.assert_equal(wm.select_chunk(policy, None, [state], 'sample4', 4, 4), baseline)

    def test_validation_uses_all_batches_and_checkpoint_roundtrip(self):
        model = WorldEnsemble(16, members=2)
        history, previous, actions = torch.zeros(5, 4, 5), torch.zeros(5, 3, 2), torch.zeros(5, 4, 2)
        targets = torch.zeros(5, 4, 5)
        targets[-1, :, 2] = .1
        model.support_bank = torch.zeros(1, 10)
        loader = DataLoader(TensorDataset(history, previous, actions, targets), batch_size=2)
        result = validate_world(model, loader, 'cpu', calibrate=True)
        self.assertEqual(result['windows'], 5)
        self.assertAlmostEqual(result['block_error_px_by_step'][-1], 5.12, places=5)
        self.assertGreater(result['loss'], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'world.pt'
            torch.save(dict(world_version=2, hidden=16, history_len=4, members=2, horizon=4, model=model.state_dict()), path)
            restored = wm.load_world(torch.device('cpu'), path)
        torch.testing.assert_close(restored.rollout(history, previous, actions), model.rollout(history, previous, actions))
        torch.testing.assert_close(restored.support_limit, model.support_limit)

    def test_reject_horizon_beyond_training(self):
        policy, world = wm.Denoiser(hidden=16), WorldEnsemble(16, horizon=4)
        with self.assertRaises(ValueError):
            wm.ChunkController(policy, world, 'world', action_steps=8)
        with self.assertRaises(ValueError):
            wm.ChunkController(policy, world, 'world', plan_steps=3)
        with self.assertRaises(ValueError):
            wm.ChunkController(policy, world, 'world', uncertainty_weight=float('nan'))


if __name__ == '__main__':
    unittest.main()
