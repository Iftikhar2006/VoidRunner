"""VoidRunner DQN network and agent utilities.

Current environment contract:
- observation shape: (18,)
- action count: 6
- action indices: brake, walk, sprint, hop, leap, reverse

This module deliberately does not modify the training loop. Phase 6 will
integrate DQNAgent with the curriculum, logs, and resumable training state.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.replay_buffer import ReplayBuffer

DEFAULT_STATE_SIZE = 18
DEFAULT_ACTION_SIZE = 6
DEFAULT_HIDDEN_SIZE = 128
CHECKPOINT_FORMAT = "voidrunner_dqn"
CHECKPOINT_FORMAT_VERSION = 3
ACTION_MASK_VERSION = "voidrunner_boundary_reverse_mask_v1"


class IncompatibleCheckpointError(ValueError):
    """Raised when a checkpoint does not match the current DQN contract."""


class DQN(nn.Module):
    """Small fully connected Q-network (online or target network)."""

    def __init__(
        self,
        state_size: int = DEFAULT_STATE_SIZE,
        action_size: int = DEFAULT_ACTION_SIZE,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        if state_size <= 0 or action_size <= 0 or hidden_size <= 0:
            raise ValueError("state_size, action_size, and hidden_size must be positive")
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.hidden_size = int(hidden_size)
        self.network = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[-1] != self.state_size:
            raise ValueError(
                f"Expected input shape (batch, {self.state_size}) or ({self.state_size},); "
                f"received {tuple(state.shape)}"
            )
        return self.network(state)


class DQNAgent:
    """Vanilla DQN: online/target networks, replay, epsilon-greedy, Huber loss."""

    def __init__(
        self,
        state_size: int = DEFAULT_STATE_SIZE,
        action_size: int = DEFAULT_ACTION_SIZE,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        replay_capacity: int = 50_000,
        min_replay_size: int = 1_000,
        target_update_frequency: int = 1_000,
        device: str | torch.device | None = None,
        seed: int = 42,
        environment_version: str | None = None,
        observation_features: Sequence[str] | None = None,
        action_names: Mapping[int | str, str] | None = None,
        permanently_masked_actions: Sequence[int] = (),
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if batch_size <= 0 or replay_capacity < batch_size:
            raise ValueError("replay_capacity must be at least batch_size > 0")
        if min_replay_size < batch_size or min_replay_size > replay_capacity:
            raise ValueError("min_replay_size must be between batch_size and replay_capacity")
        if target_update_frequency <= 0:
            raise ValueError("target_update_frequency must be positive")

        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.min_replay_size = int(min_replay_size)
        self.target_update_frequency = int(target_update_frequency)
        self.seed = int(seed)
        self.environment_version = environment_version
        self.observation_features = tuple(observation_features or ())
        self.action_names = {
            str(key): str(value) for key, value in (action_names or {}).items()
        }
        masked_actions = tuple(sorted({int(action) for action in permanently_masked_actions}))
        if any(action < 0 or action >= self.action_size for action in masked_actions):
            raise ValueError("permanently_masked_actions contains an out-of-range action index")
        if len(masked_actions) >= self.action_size:
            raise ValueError("permanently_masked_actions must leave at least one action available")
        self.permanently_masked_actions = masked_actions

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Keep the agent's exploration and replay sampling random streams distinct.
        self.rng = random.Random(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.online_network = DQN(self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.target_network = DQN(self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        self.loss_function = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
            state_size=self.state_size,
            action_size=self.action_size,
            seed=self.seed + 1,
        )
        self.learn_steps = 0

    @classmethod
    def from_env(cls, env: Any, **kwargs: Any) -> "DQNAgent":
        """Build an agent from an initialized environment's public spaces."""
        if not hasattr(env, "observation_space") or not hasattr(env, "action_space"):
            raise TypeError("env must expose Gymnasium observation_space and action_space")
        if not hasattr(env.observation_space, "shape") or env.observation_space.shape is None:
            raise ValueError("Environment observation_space must have a fixed shape")
        if len(env.observation_space.shape) != 1:
            raise ValueError("This DQN implementation expects a one-dimensional observation")
        module = importlib.import_module(env.__class__.__module__)
        kwargs.setdefault("environment_version", getattr(module, "ENVIRONMENT_VERSION", None))
        kwargs.setdefault("observation_features", getattr(module, "OBSERVATION_FEATURES", ()))
        kwargs.setdefault("action_names", getattr(module, "ACTION_NAMES", {}))
        return cls(
            state_size=int(env.observation_space.shape[0]),
            action_size=int(env.action_space.n),
            **kwargs,
        )

    def _validate_state(self, state: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(state, dtype=np.float32)
        if array.shape != (self.state_size,):
            raise ValueError(f"Expected state shape ({self.state_size},), received {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("State contains NaN or infinite values")
        return array

    def _valid_action_mask(self, state: np.ndarray) -> np.ndarray:
        """Return valid actions for a single observation.

        At the start boundary, reverse cannot move the robot farther backward
        because x is clamped at start_x. When the robot has no forward motion
        and has made no route progress, reverse is therefore masked. We do not
        mask hop/leap in the air: those commands preserve horizontal momentum
        in the current physics model and can act as a deliberate coast action.
        """
        mask = np.ones(self.action_size, dtype=bool)
        if self.state_size == 18 and self.action_size == 6:
            at_start_without_forward_motion = state[4] <= 1e-6 and state[0] <= 0.0
            if at_start_without_forward_motion:
                mask[5] = False  # reverse is unhelpful at the stationary start boundary
        for action in self.permanently_masked_actions:
            mask[action] = False
        return mask

    def _valid_action_mask_batch(self, states: torch.Tensor) -> torch.Tensor:
        """Tensor equivalent of _valid_action_mask for bootstrapped next states."""
        mask = torch.ones(
            (states.shape[0], self.action_size), dtype=torch.bool, device=states.device
        )
        if self.state_size == 18 and self.action_size == 6:
            at_start_without_forward_motion = (states[:, 4] <= 1e-6) & (states[:, 0] <= 0.0)
            mask[at_start_without_forward_motion, 5] = False
        if self.permanently_masked_actions:
            mask[:, list(self.permanently_masked_actions)] = False
        return mask

    def select_action(self, state: Sequence[float] | np.ndarray, epsilon: float = 0.0) -> int:
        """Choose a masked epsilon-greedy action; epsilon=0 is greedy evaluation."""
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        observation = self._validate_state(state)
        valid_mask = self._valid_action_mask(observation)
        valid_actions = np.flatnonzero(valid_mask)
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid_actions.tolist()))
        state_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.online_network.eval()
        with torch.no_grad():
            q_values = self.online_network(state_tensor).squeeze(0)
            mask_tensor = torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device)
            q_values = q_values.masked_fill(~mask_tensor, torch.finfo(q_values.dtype).min)
            action = int(q_values.argmax(dim=0).item())
        self.online_network.train()
        return action

    def remember(
        self,
        state: Sequence[float] | np.ndarray,
        action: int,
        reward: float,
        next_state: Sequence[float] | np.ndarray,
        terminated: bool,
        truncated: bool = False,
    ) -> None:
        """Store a transition, keeping termination separate from truncation."""
        self.replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            terminated=terminated,
            truncated=truncated,
        )

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

    def learn(self, batch_size: int | None = None) -> dict[str, float] | None:
        """Perform one DQN update; return None until enough data is available.

        True terminals stop bootstrapping. Time-limit truncations do not stop
        bootstrapping, because truncation alone does not mean the MDP ended.
        """
        size = self.batch_size if batch_size is None else int(batch_size)
        if size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self.replay_buffer) < max(size, self.min_replay_size):
            return None

        states, actions, rewards, next_states, terminated, truncated = self.replay_buffer.sample(
            size, include_truncation=True
        )
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        terminated_t = torch.as_tensor(terminated, dtype=torch.float32, device=self.device)

        self.online_network.train()
        current_q = self.online_network(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q_all = self.target_network(next_states_t)
            next_action_mask = self._valid_action_mask_batch(next_states_t)
            next_q_all = next_q_all.masked_fill(
                ~next_action_mask, torch.finfo(next_q_all.dtype).min
            )
            next_q = next_q_all.max(dim=1).values
            target_q = rewards_t + self.gamma * (1.0 - terminated_t) * next_q

        if not torch.isfinite(current_q).all() or not torch.isfinite(target_q).all():
            raise FloatingPointError("Non-finite Q-values or DQN targets detected")

        loss = self.loss_function(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.learn_steps += 1

        if self.learn_steps % self.target_update_frequency == 0:
            self.update_target_network()

        return {
            "loss": float(loss.detach().cpu().item()),
            "mean_q": float(current_q.detach().mean().cpu().item()),
            "mean_target_q": float(target_q.detach().mean().cpu().item()),
            "grad_norm": float(grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            "learn_steps": float(self.learn_steps),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        episode: int = 0,
        global_step: int = 0,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Atomically save a versioned model checkpoint (not a training resume)."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "action_mask_version": ACTION_MASK_VERSION,
            "architecture": {
                "state_size": self.state_size,
                "action_size": self.action_size,
                "hidden_size": self.hidden_size,
            },
            "environment_version": self.environment_version,
            "observation_features": list(self.observation_features),
            "action_names": dict(self.action_names),
            "agent_config": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "batch_size": self.batch_size,
                "min_replay_size": self.min_replay_size,
                "replay_capacity": self.replay_buffer.capacity,
                "target_update_frequency": self.target_update_frequency,
                "seed": self.seed,
                "device_at_save": str(self.device),
                "permanently_masked_actions": list(self.permanently_masked_actions),
            },
            "permanently_masked_actions": list(self.permanently_masked_actions),
            "episode": int(episode),
            "global_step": int(global_step),
            "learn_steps": int(self.learn_steps),
            "online_network_state_dict": self.online_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "extra_metadata": dict(extra_metadata or {}),
        }
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        expected_environment_version: str | None = None,
        load_optimizer: bool = True,
    ) -> dict[str, Any]:
        """Load only a compatible Phase 5+ checkpoint; reject legacy formats."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {source}")
        try:
            try:
                payload = torch.load(source, map_location=self.device, weights_only=True)
            except TypeError:  # compatibility with older PyTorch releases
                payload = torch.load(source, map_location=self.device)
        except Exception as exc:
            raise IncompatibleCheckpointError(
                f"Could not safely read checkpoint '{source}'. It may use an unsupported or legacy format."
            ) from exc

        if not isinstance(payload, dict) or payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
            raise IncompatibleCheckpointError(
                "This is not a versioned VoidRunner DQN checkpoint. Older checkpoints are preserved, "
                "but must not be loaded into the new 18-observation model."
            )
        if payload.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
            raise IncompatibleCheckpointError(
                f"Unsupported checkpoint format version: {payload.get('checkpoint_format_version')}"
            )
        if payload.get("action_mask_version") != ACTION_MASK_VERSION:
            raise IncompatibleCheckpointError(
                f"Action-mask policy mismatch: checkpoint={payload.get('action_mask_version')!r}, "
                f"current={ACTION_MASK_VERSION!r}. Start a new run."
            )
        stored_masked_actions = tuple(sorted({
            int(action) for action in payload.get("permanently_masked_actions", [])
        }))
        if stored_masked_actions != self.permanently_masked_actions:
            raise IncompatibleCheckpointError(
                "Permanently masked actions mismatch: "
                f"checkpoint={stored_masked_actions}, current={self.permanently_masked_actions}. "
                "Start a fresh run for a different action-mask profile."
            )
        architecture = payload.get("architecture", {})
        expected_architecture = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,
        }
        if architecture != expected_architecture:
            raise IncompatibleCheckpointError(
                f"Architecture mismatch. Checkpoint has {architecture}; current agent expects {expected_architecture}."
            )
        stored_features = tuple(payload.get("observation_features", ()))
        if self.observation_features and stored_features != self.observation_features:
            raise IncompatibleCheckpointError("Observation feature order does not match this environment.")
        stored_actions = {str(k): str(v) for k, v in payload.get("action_names", {}).items()}
        if self.action_names and stored_actions != self.action_names:
            raise IncompatibleCheckpointError("Action-index meanings do not match this environment.")
        stored_version = payload.get("environment_version")
        if expected_environment_version is not None and stored_version != expected_environment_version:
            raise IncompatibleCheckpointError(
                f"Environment version mismatch: checkpoint={stored_version!r}, "
                f"expected={expected_environment_version!r}."
            )

        try:
            self.online_network.load_state_dict(payload["online_network_state_dict"], strict=True)
            self.target_network.load_state_dict(payload["target_network_state_dict"], strict=True)
            if load_optimizer and "optimizer_state_dict" in payload:
                self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        except Exception as exc:
            raise IncompatibleCheckpointError("Checkpoint state dictionaries are incompatible with this model.") from exc
        self.learn_steps = int(payload.get("learn_steps", 0))
        self.target_network.eval()
        return {
            "episode": int(payload.get("episode", 0)),
            "global_step": int(payload.get("global_step", 0)),
            "learn_steps": self.learn_steps,
            "environment_version": stored_version,
            "extra_metadata": payload.get("extra_metadata", {}),
        }


def run_synthetic_unit_test() -> None:
    """Test network, replay learning, checkpoint save/load without Gymnasium."""
    torch.set_num_threads(1)
    agent = DQNAgent(
        state_size=18,
        action_size=6,
        hidden_size=128,
        batch_size=8,
        replay_capacity=128,
        min_replay_size=8,
        target_update_frequency=4,
        device="cpu",
        seed=2026,
        environment_version="unit_test_env",
        observation_features=[f"feature_{i}" for i in range(18)],
        action_names={i: f"action_{i}" for i in range(6)},
    )
    rng = np.random.default_rng(123)
    for i in range(32):
        state = rng.uniform(0.0, 1.0, size=18).astype(np.float32)
        state[0] = rng.uniform(-1.0, 1.0)
        next_state = rng.uniform(0.0, 1.0, size=18).astype(np.float32)
        next_state[0] = rng.uniform(-1.0, 1.0)
        terminated = (i % 11 == 0)
        truncated = (i % 7 == 0 and not terminated)
        action = agent.select_action(state, epsilon=1.0)
        agent.remember(state, action, float(rng.normal()), next_state, terminated, truncated)
    updates = [agent.learn() for _ in range(8)]
    if not any(item is not None and np.isfinite(item["loss"]) for item in updates):
        raise AssertionError("DQN did not perform a finite learning update")

    with tempfile.TemporaryDirectory(prefix="voidrunner_dqn_test_") as temp_dir:
        path = Path(temp_dir) / "smoke_checkpoint.pt"
        agent.save_checkpoint(path, episode=3, global_step=32)
        clone = DQNAgent(
            state_size=18,
            action_size=6,
            hidden_size=128,
            batch_size=8,
            replay_capacity=128,
            min_replay_size=8,
            target_update_frequency=4,
            device="cpu",
            seed=99,
            environment_version="unit_test_env",
            observation_features=[f"feature_{i}" for i in range(18)],
            action_names={i: f"action_{i}" for i in range(6)},
        )
        metadata = clone.load_checkpoint(path, expected_environment_version="unit_test_env")
        sample = np.zeros(18, dtype=np.float32)
        assert 0 <= clone.select_action(sample, epsilon=0.0) < 6
        assert metadata["episode"] == 3
        for p1, p2 in zip(agent.online_network.parameters(), clone.online_network.parameters()):
            if not torch.allclose(p1, p2):
                raise AssertionError("Reloaded network weights do not match saved weights")

        # An old-format checkpoint must be rejected, not loaded silently.
        legacy_path = Path(temp_dir) / "legacy_checkpoint.pt"
        torch.save({"model_state_dict": agent.online_network.state_dict()}, legacy_path)
        try:
            clone.load_checkpoint(legacy_path)
        except IncompatibleCheckpointError:
            pass
        else:
            raise AssertionError("Legacy checkpoint was not rejected")

    sample_batch = agent.replay_buffer.sample(8, include_truncation=True)
    expected_shapes = [(8, 18), (8,), (8,), (8, 18), (8,), (8,)]
    actual_shapes = [tuple(np.asarray(x).shape) for x in sample_batch]
    if actual_shapes != expected_shapes:
        raise AssertionError(f"Unexpected replay batch shapes: {actual_shapes}")
    print("PASS: 18-feature / 6-action network")
    print("PASS: deterministic replay-buffer sampling API and shapes")
    print("PASS: finite DQN learning updates and target-network synchronization")
    start_state = np.zeros(18, dtype=np.float32)
    if any(agent.select_action(start_state, epsilon=1.0) == 5 for _ in range(200)):
        raise AssertionError("Reverse should be masked at the stationary start boundary")
    print("PASS: reverse masked at stationary start boundary during exploration")
    masked_agent = DQNAgent(
        state_size=18, action_size=6, hidden_size=32, batch_size=8,
        replay_capacity=128, min_replay_size=8, target_update_frequency=4,
        device="cpu", seed=2027, environment_version="unit_test_env",
        permanently_masked_actions=(5,),
    )
    nonstart_state = np.zeros(18, dtype=np.float32)
    nonstart_state[4] = 0.2
    selected = [masked_agent.select_action(nonstart_state, epsilon=1.0) for _ in range(500)]
    if any(action == 5 for action in selected):
        raise AssertionError("A permanently masked action was selected during exploration")
    batch_mask = masked_agent._valid_action_mask_batch(torch.zeros((4, 18), dtype=torch.float32))
    if bool(batch_mask[:, 5].any()):
        raise AssertionError("A permanently masked action remains valid in TD targets")
    mask_path = Path(temp_dir) / "permanently_masked_checkpoint.pt"
    masked_agent.save_checkpoint(mask_path, episode=1, global_step=10)
    mask_clone = DQNAgent(
        state_size=18, action_size=6, hidden_size=32, batch_size=8,
        replay_capacity=128, min_replay_size=8, target_update_frequency=4,
        device="cpu", seed=2028, environment_version="unit_test_env",
        permanently_masked_actions=(5,),
    )
    mask_clone.load_checkpoint(mask_path, expected_environment_version="unit_test_env")
    try:
        clone.load_checkpoint(mask_path, expected_environment_version="unit_test_env")
    except IncompatibleCheckpointError:
        pass
    else:
        raise AssertionError("Checkpoint with a different action-mask profile was accepted")
    print("PASS: permanent action mask applied to exploration, greedy choice, and TD targets")
    print("PASS: versioned checkpoint save/load with action-mask metadata")
    print("PASS: legacy checkpoint rejection")


def run_environment_smoke_test(output_path: Path) -> None:
    """Exercise the real Gymnasium environment and save a smoke checkpoint."""
    try:
        from environment.terrain_env import (
            ACTION_NAMES,
            ENVIRONMENT_VERSION,
            OBSERVATION_FEATURES,
            TerrainEnv,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not import the actual TerrainEnv. Run this command inside the project's "
            "voidrunner conda environment with Gymnasium installed."
        ) from exc

    torch.set_num_threads(1)
    env = TerrainEnv(difficulty="easy", render_mode=None, max_steps=200)
    try:
        agent = DQNAgent.from_env(
            env,
            hidden_size=128,
            learning_rate=5e-4,
            gamma=0.99,
            batch_size=8,
            replay_capacity=256,
            min_replay_size=8,
            target_update_frequency=10,
            seed=2026,
        )
        if agent.state_size != 18 or agent.action_size != 6:
            raise AssertionError(
                f"Expected 18 observations and 6 actions, received {agent.state_size} and {agent.action_size}"
            )
        state, info = env.reset(seed=2026)
        learned_updates = 0
        total_steps = 0
        episode_count = 1
        last_metrics: dict[str, float] | None = None
        while total_steps < 48:
            action = agent.select_action(state, epsilon=1.0)
            next_state, reward, terminated, truncated, info = env.step(action)
            if not env.observation_space.contains(next_state):
                raise AssertionError("Environment produced an invalid next observation")
            agent.remember(state, action, reward, next_state, terminated, truncated)
            last_metrics = agent.learn()
            if last_metrics is not None:
                learned_updates += 1
            state = next_state
            total_steps += 1
            if terminated or truncated:
                episode_count += 1
                state, info = env.reset(seed=2026 + episode_count - 1)
        if learned_updates == 0:
            raise AssertionError("No DQN learning updates occurred during the environment smoke test")
        saved_path = agent.save_checkpoint(
            output_path,
            episode=episode_count,
            global_step=total_steps,
            extra_metadata={"purpose": "phase5_environment_smoke_test", "terrain_seed_initial": 2026},
        )
        fresh_agent = DQNAgent.from_env(
            env,
            hidden_size=128,
            learning_rate=5e-4,
            gamma=0.99,
            batch_size=8,
            replay_capacity=256,
            min_replay_size=8,
            target_update_frequency=10,
            seed=9,
        )
        loaded = fresh_agent.load_checkpoint(
            saved_path,
            expected_environment_version=ENVIRONMENT_VERSION,
        )
        print(f"Environment version: {ENVIRONMENT_VERSION}")
        print(f"Observation features: {len(OBSERVATION_FEATURES)}")
        print(f"Actions: {len(ACTION_NAMES)}")
        print(f"Environment steps: {total_steps}; episodes touched: {episode_count}")
        print(f"Successful learning updates: {learned_updates}")
        print(f"Last update metrics: {last_metrics}")
        print(f"Saved and reloaded checkpoint: {saved_path}")
        print(f"Checkpoint metadata: {loaded}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="VoidRunner DQN Phase 5 tests")
    parser.add_argument(
        "--unit-test",
        action="store_true",
        help="test the network, replay buffer, learning update, and checkpoint without Gymnasium",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run a short DQN interaction test against the real TerrainEnv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase5_smoke_test/dqn_smoke_test.pt"),
        help="output checkpoint path for --smoke-test",
    )
    args = parser.parse_args()
    if args.unit_test == args.smoke_test:
        parser.error("Choose exactly one of --unit-test or --smoke-test")
    if args.unit_test:
        run_synthetic_unit_test()
    else:
        run_environment_smoke_test(args.output)


if __name__ == "__main__":
    main()
